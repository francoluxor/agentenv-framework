"""Offline unit tests for the Tenki sandbox provider."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tenki_sandbox.models import SessionEgressPolicy

from agent_env.config.errors import ConfigError
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy, NetworkPolicyUnsupportedError
from agent_env.providers.sandbox_providers.tenki import provider as provider_module
from agent_env.providers.sandbox_providers.tenki.provider import (
    DEFAULT_BASE_URL,
    DEFAULT_MIN_DISK_GB,
    SESSION_TAG,
    TenkiSandboxProvider,
    egress_kwargs,
    session_name,
    vm_shape,
)
from agent_env.providers.sandbox_providers.tenki.sandbox import DOCKER_BOOTSTRAP, TenkiSandbox
from tst.unit.providers.sandbox_providers.tenki_fakes import FakeSession, fake_client_cls, result

ALLOWLIST = NetworkMode.ALLOWLIST


@pytest.fixture(autouse=True)
def _no_preview_parents(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(provider_module, "_preview_parents", set())


@pytest.fixture
def client_cls():
    return fake_client_cls()


def _provider(client_cls, **kwargs) -> TenkiSandboxProvider:
    return TenkiSandboxProvider(api_key="tk_test", client_cls=client_cls, **kwargs)


def test_vm_shape_rounds_up_to_tenki_units():
    assert vm_shape(2.1, 8193, 42.2, min_disk_gb=DEFAULT_MIN_DISK_GB) == (3, 8194, 43)
    assert vm_shape(4, 8192, 10, min_disk_gb=50) == (4, 8192, 50)


def test_vm_shape_raises_a_small_request_to_the_warm_start_floor():
    """A VM smaller than 2 vCPU / 4 GiB cold boots instead of restoring Tenki's warm snapshot."""
    assert vm_shape(0.5, 1001, 3, min_disk_gb=DEFAULT_MIN_DISK_GB) == (2, 4096, DEFAULT_MIN_DISK_GB)
    assert vm_shape(1, 8192, 10, min_disk_gb=DEFAULT_MIN_DISK_GB) == (2, 8192, DEFAULT_MIN_DISK_GB)


@pytest.mark.parametrize(("cpu", "memory", "disk"), [(129, 1024, 10), (1, 524290, 10), (1, 1024, 101)])
def test_vm_shape_refuses_rather_than_shrinks(cpu, memory, disk):
    with pytest.raises(ValueError, match="Tenki cannot size a VM"):
        vm_shape(cpu, memory, disk, min_disk_gb=DEFAULT_MIN_DISK_GB)


def test_egress_allow_all_sends_nothing():
    assert egress_kwargs(NetworkPolicy()) == {}


def test_egress_allowlist_splits_hosts_and_cidrs():
    policy = NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org", "10.1.2.3"), allow_cidrs=("192.168.0.0/16",))
    assert egress_kwargs(policy) == {"allow_domains": ["pypi.org"], "allow_cidrs": ["192.168.0.0/16", "10.1.2.3/32"]}


def test_an_empty_allowlist_turns_outbound_off():
    """Tenki reads empty allowlists as unrestricted, which would open egress."""
    assert egress_kwargs(NetworkPolicy(mode=ALLOWLIST)) == {"allow_outbound": False}


def test_supports_allow_all_and_ipv4_allowlists_within_tenki_limits():
    assert TenkiSandboxProvider.supports_network_policy(NetworkPolicy())
    assert TenkiSandboxProvider.supports_network_policy(NetworkPolicy(mode=ALLOWLIST, allow_hosts=("a.com",)))
    assert not TenkiSandboxProvider.supports_network_policy(
        NetworkPolicy(mode=ALLOWLIST, allow_cidrs=("2001:db8::/32",))
    )
    too_many = tuple(f"h{i}.example.com" for i in range(257))
    assert not TenkiSandboxProvider.supports_network_policy(NetworkPolicy(mode=ALLOWLIST, allow_hosts=too_many))


def test_session_name_fits_tenki_limits():
    name = session_name({"run_id": "Run 42/" + "x" * 80, "team": "Env Pod"})
    assert name.startswith("agentenv-")
    assert len(name) <= 63
    assert name == name.lower() and " " not in name


def test_from_config_takes_the_known_keys():
    provider = TenkiSandboxProvider.from_config(api_key="tk_x", snapshot_id="snap", min_disk_gb=40)
    assert (provider._api_key, provider._snapshot_id, provider._min_disk_gb) == ("tk_x", "snap", 40)
    assert provider._base_url == DEFAULT_BASE_URL


@pytest.mark.parametrize(("config", "message"), [
    ({"api_key": "k", "region": "us"}, "unknown key"),
    ({}, "requires a non-empty 'api_key'"),
    ({"api_key": "k", "base_url": " "}, "non-empty string"),
    ({"api_key": "k", "image": "a", "snapshot_id": "b"}, "at most one"),
    ({"api_key": "k", "min_disk_gb": 2}, "min_disk_gb"),
    ({"api_key": "k", "min_disk_gb": True}, "min_disk_gb"),
])
def test_from_config_refuses_bad_config(config, message):
    with pytest.raises(ConfigError, match=message):
        TenkiSandboxProvider.from_config(**config)


@pytest.mark.asyncio
async def test_create_vm_sizes_tags_and_exposes_ports(client_cls):
    sandbox = await _provider(client_cls, snapshot_id="snap").create_vm(
        cpu=1.5, memory=4096, disk_size_gb=10, timeout=900, exposed_ports=[8080, 8080, 9000],
        setup_for_gateway=False, attribution={"run_id": "r1"},
    )
    client = client_cls.instances[0]
    assert client.kwargs == {"api_key": "tk_test", "base_url": DEFAULT_BASE_URL}
    call = client.create_calls[0]
    assert (call["cpu_cores"], call["memory_mb"], call["disk_size_gb"]) == (2, 4096, DEFAULT_MIN_DISK_GB)
    assert call["max_duration"] == 900
    assert call["tags"] == [SESSION_TAG]
    assert call["snapshot_id"] == "snap"
    assert call["metadata"]["run_id"] == "r1"
    assert call["wait"] is False
    assert "allow_domains" not in call and "allow_outbound" not in call
    assert isinstance(sandbox, TenkiSandbox)
    assert (sandbox.type, sandbox.mode) == ("tenki", "vm")
    assert sandbox.tunnel_urls == {8080: "https://p8080-sess-1.sb.tenki.sh", 9000: "https://p9000-sess-1.sb.tenki.sh"}
    assert sandbox._command_timeout == 900


@pytest.mark.asyncio
async def test_create_vm_applies_an_allowlist_with_the_tenki_hosts(client_cls):
    sandbox = await _provider(client_cls).create_vm(
        setup_for_gateway=False, network_policy=NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org",)),
    )
    domains = client_cls.instances[0].create_calls[0]["allow_domains"]
    assert {"pypi.org", "*.sb.tenki.sh", "*.us.sb.tenki.sh"} <= set(domains)
    assert sandbox.network_policy.restricts_egress


@pytest.mark.asyncio
async def test_workspace_preview_domains_join_later_allowlists(client_cls):
    def wildcard_workspace(client):
        client.next_session = FakeSession("env-1")
        client.next_session.preview_parent = "acme.us.sb.tenki.sh"

    provider = _provider(client_cls)
    client_cls.on_init = wildcard_workspace
    env = await provider.create_vm(setup_for_gateway=False, exposed_ports=[8080])
    assert env.tunnel_urls[8080] == "https://p8080-env-1.acme.us.sb.tenki.sh"

    client_cls.on_init = None
    await provider.create_vm(
        setup_for_gateway=False, network_policy=NetworkPolicy(mode=ALLOWLIST, allow_hosts=("a.io",)),
    )
    assert "*.acme.us.sb.tenki.sh" in client_cls.instances[-1].create_calls[0]["allow_domains"]


@pytest.mark.parametrize(("url", "learned"), [
    ("https://tok.acme.us.sb.tenki.sh", {"*.acme.us.sb.tenki.sh"}),
    ("https://tok.sb.tenki.sh", set()),
    ("https://tok.us.sb.tenki.sh", set()),
    ("https://tok.evil-sb.tenki.sh", set()),
    ("https://tok.example.com", set()),
])
def test_only_tenki_preview_parents_are_learned(url, learned):
    provider_module._tunnel_url(url)
    assert provider_module._preview_parents == learned


def test_learned_preview_parents_are_capped():
    for i in range(provider_module._MAX_PREVIEW_PARENTS + 5):
        provider_module._tunnel_url(f"https://tok.ws{i}.us.sb.tenki.sh")
    assert len(provider_module._preview_parents) == provider_module._MAX_PREVIEW_PARENTS


@pytest.mark.asyncio
async def test_create_vm_refuses_an_unenforceable_policy_before_creating(client_cls):
    with pytest.raises(NetworkPolicyUnsupportedError):
        await _provider(client_cls).create_vm(network_policy=NetworkPolicy(mode=ALLOWLIST, allow_cidrs=("2001:db8::/32",)))
    assert client_cls.instances == []


@pytest.mark.asyncio
async def test_create_vm_refuses_a_per_call_image(client_cls):
    with pytest.raises(ValueError, match="per-call image"):
        await _provider(client_cls).create_vm(image="ubuntu:24.04")
    assert client_cls.instances == []


@pytest.mark.asyncio
async def test_create_vm_readies_docker_for_the_gateway_as_root(client_cls):
    await _provider(client_cls).create_vm(exposed_ports=[8080])
    session = client_cls.instances[0].sessions["sess-1"]
    argvs = [call["argv"] for call in session.exec_calls]
    assert ["bash", "-c", DOCKER_BOOTSTRAP] in argvs
    assert ["docker", "compose", "version"] in argvs
    assert all(call["privileged"] for call in session.exec_calls)


@pytest.mark.asyncio
async def test_a_setup_failure_terminates_the_vm_and_closes_the_client(client_cls):
    session = FakeSession()
    session.results = [result(stderr=b"no space left", exit_code=1)]
    client_cls.on_init = lambda client: setattr(client, "next_session", session)
    with pytest.raises(RuntimeError, match="no space left"):
        await _provider(client_cls).create_vm()
    assert session.closed == 1
    assert client_cls.instances[0].closed


@pytest.mark.asyncio
async def test_a_cancelled_create_reaps_the_vm(client_cls):
    client_cls.on_init = lambda client: setattr(client, "create_delay", 0.05)
    task = asyncio.ensure_future(_provider(client_cls).create_vm(setup_for_gateway=False))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    client = client_cls.instances[0]
    assert not client.closed, "the reaper still needs the client to terminate the VM"
    await asyncio.sleep(0.1)
    assert client.sessions["sess-1"].closed == 1
    assert client.closed


@pytest.mark.asyncio
async def test_a_cancelled_create_that_then_fails_still_closes_the_client(client_cls):
    def failing_slow_create(client):
        client.create_delay = 0.05
        client.create_error = RuntimeError("capacity unavailable")

    client_cls.on_init = failing_slow_create
    task = asyncio.ensure_future(_provider(client_cls).create_vm(setup_for_gateway=False))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.1)
    assert client_cls.instances[0].closed


@pytest.mark.asyncio
async def test_a_failed_create_closes_the_client(client_cls):
    client_cls.on_init = lambda client: setattr(client, "create_error", RuntimeError("capacity unavailable"))
    with pytest.raises(RuntimeError, match="capacity unavailable"):
        await _provider(client_cls).create_vm(setup_for_gateway=False)
    assert client_cls.instances[0].closed


@pytest.mark.asyncio
async def test_create_sandbox_is_a_vm_with_its_port(client_cls):
    sandbox = await _provider(client_cls).create_sandbox(image_name="agent:latest", port=8000, env={"K": "V"})
    assert list(sandbox.tunnel_urls) == [8000]
    assert "env" not in client_cls.instances[0].create_calls[0]


@pytest.mark.asyncio
async def test_get_sandbox_restores_ports_policy_and_lifetime(client_cls):
    session = FakeSession(
        "sess-9", state="RUNNING",
        egress=SessionEgressPolicy(allow_domains=("pypi.org",), allow_cidrs=("10.0.0.0/8",)),
        timeout_at=datetime.now(UTC) + timedelta(minutes=30),
    )
    session.exposed = {8080: "p8080.sb.tenki.sh"}
    client_cls.on_init = lambda client: client.sessions.update({"sess-9": session})
    sandbox = await _provider(client_cls).get_sandbox("sess-9")
    assert sandbox.tunnel_urls == {8080: "https://p8080.sb.tenki.sh"}
    assert sandbox.network_policy == NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org",), allow_cidrs=("10.0.0.0/8",))
    assert 1700 <= sandbox._command_timeout <= 1800
    await sandbox.terminate()
    assert client_cls.instances[0].closed
