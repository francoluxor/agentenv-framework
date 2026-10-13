"""Tenki Sandbox VM sandbox provider.

Each sandbox is one Tenki session, a full Linux VM. The gateway and agent containers run under Docker inside it
through the inherited ``VmSandbox`` helpers; a VM booted from an image without Docker installs it first.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import math
import re
import uuid
from datetime import UTC, datetime
from typing import Any, ClassVar, Self
from urllib.parse import urlparse

from tenki import AsyncClient

from agent_env.attribution import Attribution
from agent_env.config.errors import ConfigError
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy, NetworkPolicyUnsupportedError
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider, apply_default_attribution
from agent_env.providers.sandbox_providers.tenki.sandbox import TenkiSandbox, policy_from_info

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.tenki.cloud"
SESSION_TAG = "agentenv"
_SECTION = "[sandbox.providers.tenki.config]"
_STRING_KEYS = ("api_key", "base_url", "image", "snapshot_id")
_CONFIG_KEYS = frozenset({*_STRING_KEYS, "min_disk_gb"})

# Tenki's API bounds; a workspace's own limit may be lower and is enforced at create.
# Tenki restores VMs of at least this size from a warm snapshot; a smaller one cold boots, which is slower.
MIN_CPUS = 2
MIN_MEMORY_MB = 4096
MAX_CPUS = 128
MAX_MEMORY_MB = 512 * 1024
MIN_DISK_GB = 5
MAX_DISK_GB = 100
# Docker keeps both the compressed and the extracted layers, so 10 GB runs out once a gateway and a few MCP server
# images are on the VM.
DEFAULT_MIN_DISK_GB = 20
MAX_ALLOW_DOMAINS = 256
MAX_ALLOW_CIDRS = 64

READY_TIMEOUT_SEC = 300
_REAP_ATTEMPTS = 3
_reapers: set[asyncio.Task] = set()

# A workspace with wildcard preview domains gets URLs two labels under the regional suffix
# (<token>.<workspace>.us.sb.tenki.sh), which a one-label wildcard in EGRESS_HOSTS can't cover. The parents seen on
# exposed ports are added to later allowlists; the API has no call that lists them up front.
_preview_parents: set[str] = set()


def vm_shape(cpu: float, memory_mb: int, disk_size_gb: float, *, min_disk_gb: int) -> tuple[int, int, int]:
    """``(cpu_cores, memory_mb, disk_size_gb)``: whole cores and memory rounded up to Tenki's 2 MiB alignment, raised to
    the warm-start floor, and a disk of at least ``min_disk_gb``. A request beyond Tenki's bounds is refused, not
    shrunk."""
    cores = max(MIN_CPUS, math.ceil(cpu))
    memory = max(MIN_MEMORY_MB, math.ceil(memory_mb / 2) * 2)
    disk = max(MIN_DISK_GB, min_disk_gb, math.ceil(disk_size_gb))
    if cores > MAX_CPUS or memory > MAX_MEMORY_MB or disk > MAX_DISK_GB:
        raise ValueError(
            f"Tenki cannot size a VM for cpu={cpu}, memory={memory_mb}MB, disk={disk_size_gb}GB "
            f"(at most {MAX_CPUS} vCPU, {MAX_MEMORY_MB} MB memory, {MAX_DISK_GB} GB disk)"
        )
    return cores, memory, disk


def _split_entries(policy: NetworkPolicy) -> tuple[list[str], list[str]]:
    """Tenki's ``allow_domains`` and ``allow_cidrs``; an IP address listed as a host becomes a /32."""
    domains: list[str] = []
    cidrs = list(policy.allow_cidrs)
    for host in policy.allow_hosts:
        try:
            cidrs.append(str(ipaddress.ip_network(host, strict=False)))
        except ValueError:
            domains.append(host)
    return domains, cidrs


def egress_kwargs(policy: NetworkPolicy) -> dict[str, Any]:
    """``AsyncClient.create`` egress arguments for ``policy``. Tenki reads empty allowlists as unrestricted, so an
    allowlist with no entries turns outbound off instead."""
    if not policy.restricts_egress:
        return {}
    domains, cidrs = _split_entries(policy)
    if not domains and not cidrs:
        return {"allow_outbound": False}
    return {"allow_domains": domains or None, "allow_cidrs": cidrs or None}


def session_name(attribution: Attribution) -> str:
    """``agentenv-<random>`` and the attribution values, slugged into Tenki's 63-character name."""
    slugs = [re.sub(r"[^a-z0-9]+", "-", str(attribution[key]).lower()).strip("-") for key in sorted(attribution)]
    return "-".join(["agentenv", uuid.uuid4().hex[:8], *filter(None, slugs)])[:63].rstrip("-")


def _tunnel_url(url: str) -> str:
    url = url if url.startswith(("http://", "https://")) else f"https://{url}"
    parent = (urlparse(url).hostname or "").partition(".")[2]
    if parent.endswith("sb.tenki.sh") and f"*.{parent}" not in TenkiSandboxProvider.EGRESS_HOSTS:
        _preview_parents.add(f"*.{parent}")
    return url


def _remaining_lifetime(timeout_at: datetime | None) -> int:
    if timeout_at is None:
        return 2 * 3600
    if timeout_at.tzinfo is None:
        timeout_at = timeout_at.replace(tzinfo=UTC)
    return max(60, math.ceil((timeout_at - datetime.now(UTC)).total_seconds()))


async def _reap(sandbox: TenkiSandbox) -> None:
    for attempt in range(_REAP_ATTEMPTS):
        try:
            await sandbox.terminate()
            logger.info("Terminated Tenki VM %s, created after its caller was cancelled", sandbox.sandbox_id)
            return
        except Exception as exc:  # noqa: BLE001 - retried, then logged
            logger.warning("Terminating orphaned Tenki VM %s failed (attempt %s): %s", sandbox.sandbox_id,
                           attempt + 1, exc)
            await asyncio.sleep(2**attempt)
    logger.error("Orphaned Tenki VM %s is still running; it stops at its max lifetime", sandbox.sandbox_id)


class TenkiSandboxProvider(SandboxProvider):
    """Provision Docker-capable Tenki VMs.

    ``api_key`` comes from resolved provider config, never from the process environment: configure it with an
    ``env:`` or ``secret:`` reference in ``[sandbox.providers.tenki.config]``. Workloads never see it.
    """

    # Exposed ports get public URLs under these domains. Tenki wildcards cover one label, hence the regional entry.
    EGRESS_HOSTS: ClassVar[tuple[str, ...]] = ("*.sb.tenki.sh", "*.us.sb.tenki.sh")

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        image: str | None = None,
        snapshot_id: str | None = None,
        min_disk_gb: int = DEFAULT_MIN_DISK_GB,
        client_cls: Any = AsyncClient,
    ):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Tenki api_key must be a non-empty resolved secret")
        if image and snapshot_id:
            raise ValueError("Pass at most one of image and snapshot_id")
        self._api_key = api_key
        self._base_url = base_url
        self._image = image
        self._snapshot_id = snapshot_id
        self._min_disk_gb = min_disk_gb
        # Injectable so unit tests never build an SDK client.
        self._client_cls = client_cls

    @classmethod
    def from_config(cls, **config: Any) -> Self:
        unknown = set(config) - _CONFIG_KEYS
        if unknown:
            raise ConfigError(f"{_SECTION} has unknown key(s): {sorted(unknown)}")
        if not config.get("api_key"):
            raise ConfigError(f"{_SECTION} requires a non-empty 'api_key'")
        for key in _STRING_KEYS:
            value = config.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(f"{_SECTION} '{key}' must be a non-empty string")
        if config.get("image") and config.get("snapshot_id"):
            raise ConfigError(f"{_SECTION} takes at most one of 'image' and 'snapshot_id'")
        min_disk_gb = config.get("min_disk_gb", DEFAULT_MIN_DISK_GB)
        if isinstance(min_disk_gb, bool) or not isinstance(min_disk_gb, int) or not (
            MIN_DISK_GB <= min_disk_gb <= MAX_DISK_GB
        ):
            raise ConfigError(f"{_SECTION} 'min_disk_gb' must be an integer from {MIN_DISK_GB} to {MAX_DISK_GB}")
        return cls(**config)

    @classmethod
    def effective_network_policy(cls, policy: NetworkPolicy | None) -> NetworkPolicy:
        """Every platform's hosts, as on the base, and the workspace preview domains seen so far."""
        return super().effective_network_policy(policy).with_hosts(sorted(_preview_parents))

    @classmethod
    def supports_network_policy(cls, policy: NetworkPolicy) -> bool:
        """Allow-all, or an allowlist of hostnames and IPv4 addresses or CIDRs within Tenki's limits."""
        if not policy.restricts_egress:
            return True
        domains, cidrs = _split_entries(policy)
        return (
            len(domains) <= MAX_ALLOW_DOMAINS
            and len(cidrs) <= MAX_ALLOW_CIDRS
            and all(ipaddress.ip_network(cidr, strict=False).version == 4 for cidr in cidrs)
        )

    def _client(self) -> Any:
        return self._client_cls(api_key=self._api_key, base_url=self._base_url)

    async def create_vm(
        self,
        *,
        image: str | None = None,
        boot_mode: str | None = None,
        cpu: float = 1.0,
        memory: int = 8192,
        disk_size_gb: float = 10,
        timeout: int = 3600 * 2,
        exposed_ports: list[int] | None = None,
        setup_for_gateway: bool = True,
        attribution: Attribution | None = None,
        network_policy: NetworkPolicy | None = None,
    ) -> TenkiSandbox:
        """Create a Tenki VM and expose ``exposed_ports`` publicly. ``timeout`` is the VM's hard maximum lifetime. The
        VM boots the configured image or snapshot, so a per-call ``image`` is refused rather than ignored;
        ``boot_mode`` has no Tenki equivalent and is ignored."""
        if image is not None:
            raise ValueError(
                "Tenki VMs boot the image or snapshot_id in [sandbox.providers.tenki.config]; "
                "a per-call image is unsupported"
            )
        if boot_mode is not None:
            logger.warning("Ignoring boot_mode=%s: Tenki VMs have no boot modes", boot_mode)
        cores, memory_mb, disk_gb = vm_shape(cpu, memory, disk_size_gb, min_disk_gb=self._min_disk_gb)
        effective_policy = self.effective_network_policy(network_policy)
        if not self.supports_network_policy(effective_policy):
            raise NetworkPolicyUnsupportedError(
                f"Tenki enforces allow-all or an allowlist of up to {MAX_ALLOW_DOMAINS} hostnames and "
                f"{MAX_ALLOW_CIDRS} IPv4 CIDRs, not {effective_policy.to_dict()}"
            )
        metadata = {
            key: str(value)
            for key, value in apply_default_attribution(dict(attribution or {})).items()
            if value is not None
        }
        ports = list(dict.fromkeys(exposed_ports or []))
        create_kwargs: dict[str, Any] = {
            **egress_kwargs(effective_policy),
            "name": session_name(metadata),
            "cpu_cores": cores,
            "memory_mb": memory_mb,
            "disk_size_gb": disk_gb,
            "max_duration": timeout,
            "metadata": metadata,
            "tags": [SESSION_TAG],
            "wait": False,
        }
        if self._image:
            create_kwargs["image"] = self._image
        if self._snapshot_id:
            create_kwargs["snapshot_id"] = self._snapshot_id

        client = self._client()
        try:
            session = await self._create_or_reclaim(client, create_kwargs, effective_policy, timeout)
        except BaseException:
            await client.close()
            raise
        sandbox = TenkiSandbox(
            session, client=client, tunnel_urls={}, network_policy=effective_policy, command_timeout=timeout,
        )
        try:
            await session.wait_ready(timeout=READY_TIMEOUT_SEC)
            sandbox.tunnel_urls = await self._expose(session, ports)
            logger.info(
                "Tenki VM started: session_id=%s cpu=%s memory=%sMB disk=%sGB ports=%s attribution=%s",
                session.id, cores, memory_mb, disk_gb, ports, metadata,
            )
            if setup_for_gateway:
                await sandbox.setup_vm_for_gateway(ports)
            return sandbox
        except BaseException:
            try:
                await sandbox.terminate()
            except Exception as cleanup_error:  # noqa: BLE001 - cleanup must not mask the create failure
                logger.warning("Failed to terminate Tenki VM %s after setup failure: %s", session.id, cleanup_error)
            raise

    @staticmethod
    async def _create_or_reclaim(
        client: Any, create_kwargs: dict[str, Any], policy: NetworkPolicy, timeout: int
    ) -> Any:
        """Await the create. If the caller is cancelled first, terminate the VM it yields, which would otherwise run
        with no handle until its max lifetime."""
        task = asyncio.ensure_future(client.create(**create_kwargs))

        def terminate_orphan(done: asyncio.Future) -> None:
            if done.cancelled() or done.exception() is not None:
                return
            orphan = TenkiSandbox(
                done.result(), client=client, tunnel_urls={}, network_policy=policy, command_timeout=timeout,
            )
            reaper = asyncio.ensure_future(_reap(orphan))
            _reapers.add(reaper)
            reaper.add_done_callback(_reapers.discard)

        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            task.add_done_callback(terminate_orphan)
            raise

    @staticmethod
    async def _expose(session: Any, ports: list[int]) -> dict[int, str]:
        exposed = await asyncio.gather(*(session.expose_port(port) for port in ports))
        return {port: _tunnel_url(item.url) for port, item in zip(ports, exposed, strict=True)}

    async def create_sandbox(
        self,
        *,
        image_name: str,
        port: int,
        env: dict[str, str],
        cpu: float = 1.0,
        memory: int = 8192,
        disk_size_gb: float = 10,
        timeout: int = 3600 * 2,
        attribution: Attribution | None = None,
        network_policy: NetworkPolicy | None = None,
    ) -> TenkiSandbox:
        # A VM backend: the gateway and agent paths start image_name inside the VM afterwards.
        del image_name, env
        return await self.create_vm(
            cpu=cpu,
            memory=memory,
            disk_size_gb=disk_size_gb,
            timeout=timeout,
            exposed_ports=[port],
            attribution=attribution,
            network_policy=network_policy,
        )

    async def get_sandbox(self, sandbox_id: str) -> TenkiSandbox:
        """Reconnect, restoring the exposed ports and the applied egress policy."""
        client = self._client()
        try:
            session = await client.get(sandbox_id)
            exposed = await session.list_exposed_ports()
        except BaseException:
            await client.close()
            raise
        info = session.info
        return TenkiSandbox(
            session,
            client=client,
            tunnel_urls={item.port: _tunnel_url(item.url) for item in exposed},
            network_policy=policy_from_info(info),
            command_timeout=_remaining_lifetime(info.timeout_at),
        )
