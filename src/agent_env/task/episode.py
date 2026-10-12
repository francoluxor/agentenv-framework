"""Run a task with some of its agents played from outside AgentEnv, by a training loop's model or any other caller.

AgentEnv runs everything else exactly as ``Task.run`` does, so the reward is the score ``agent-env run`` would give.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from agent_env.entity_refs import EntityRef
from agent_env.task.task import Task, substitute_seed_in_steps
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep

_AGENT_DEPLOYERS = ("install_agent", "deploy_human_agent")
_AGENT_TARGETING = (
    "add_skills", "register_agent_triggers", "snapshot_agent_state", "load_artifact", "verify_sandbox", "run_code",
)


class EpisodeError(ValueError):
    """A task that can't run as an episode, or an episode asked for what it can't give."""


@dataclass(frozen=True)
class TurnSpec:
    step_id: str
    prompt: str
    system_prompt: str | None


@dataclass(frozen=True)
class AgentSpec:
    role: str | None
    turns: tuple[TurnSpec, ...]


class EpisodeSpec:
    """A task read as an episode: the agents played from outside, their turns, and the copy of the task that runs."""

    def __init__(
        self, *, copy: Task, agents: dict[str, AgentSpec], reward_verifier_id: str, seed: dict[str, str] | None,
    ):
        self._copy = copy
        self.agents = agents
        self._reward_verifier_id = reward_verifier_id
        self._seed = seed

    @classmethod
    def from_task(
        cls, task: Task, *, agents: Sequence[str] | None = None, seed: Mapping[str, str] | None = None,
    ) -> EpisodeSpec:
        seed = dict(seed) if seed else None
        seeded = substitute_seed_in_steps(task.steps, seed or {})
        outside = _outside_agents(task, seeded, agents)
        problems = _problems(seeded, outside)
        verifiers = _top_level_verifiers(seeded)
        if len(verifiers) != 1:
            problems.append(
                f"the reward is the score of the one verifier no aggregate_verifiers step lists, and the task has "
                f"{len(verifiers)}" + (f": {[step.id for step in verifiers]}" if verifiers else "")
            )
        if problems:
            raise EpisodeError(f"task {task.id!r} can't run as an episode: " + "; ".join(problems))
        deploys = {
            step.agent_name: step for step in task.steps if step.type == "deploy_agent" and step.agent_name in outside
        }
        return cls(
            copy=Task(id=task.id, version=task.version, steps=[_stand_in(step, deploys) for step in task.steps]),
            agents={name: _agent_spec(seeded, name) for name in outside},
            reward_verifier_id=verifiers[0].verifier_id,
            seed=seed,
        )


def _outside_agents(task: Task, steps: list[TaskStep], agents: Sequence[str] | None) -> list[str]:
    prompted = list(dict.fromkeys(step.agent_name for step in steps if step.type == "prompt_agent"))
    if agents is None:
        if len(prompted) != 1:
            raise EpisodeError(
                f"task {task.id!r} prompts {prompted}; name the ones to play with agents=" if prompted
                else f"task {task.id!r} prompts no agent"
            )
        return prompted
    outside = list(dict.fromkeys(agents))
    if not outside:
        raise EpisodeError(f"agents= names no agent; task {task.id!r} prompts {prompted}")
    if unprompted := [name for name in outside if name not in prompted]:
        raise EpisodeError(f"task {task.id!r} doesn't prompt {unprompted}; it prompts {prompted}")
    return outside


def _problems(steps: list[TaskStep], outside: list[str]) -> list[str]:
    """Why each step, and each outside agent, can't be run with the agent played from outside."""
    problems = []
    for name in outside:
        deploys = [step for step in steps if step.type == "deploy_agent" and step.agent_name == name]
        if len(deploys) != 1:
            problems.append(f"agent {name!r} is played from outside, so it needs exactly one deploy_agent step, "
                            f"not {len(deploys)}")
    outside_prompts = {
        step.id: step.prompt_id for step in steps if step.type == "prompt_agent" and step.agent_name in outside
    }
    for step in steps:
        problems.extend(f"step {step.id!r}: {why}" for why in _step_problems(step, outside, outside_prompts))
    return problems


def _step_problems(step: TaskStep, outside: list[str], outside_prompts: dict[str, str]) -> list[str]:
    why = []
    if step.retry_config is not None:
        why.append("retry_config can replay an outside turn")
    agent = getattr(step, "agent_name", None)
    if step.type == "prompt_agent" and agent in outside:
        why.extend(
            f"an outside agent's prompt can't have {what}" for what, present in (
                ("parts, only a text prompt", step.parts is not None),
                ("a user simulator (max_conversation_turns > 1)", step.max_conversation_turns > 1),
                ("output_format", step.output_format),
                ("agentenv_tools", step.agentenv_tools),
                ("snapshot_config", step.snapshot_config),
            ) if present
        )
    elif step.type == "prompt_agent" and step.max_conversation_turns > 1 and step.user_agent_name in outside:
        why.append(f"its user simulator {step.user_agent_name!r} is played from outside")
    elif step.type == "deploy_agent" and agent in outside:
        why.extend(
            f"an outside agent can't have {what}" for what, present in (
                ("skills, instructions its player wouldn't get", step.skills),
                ("agent_snapshot_files_artifact_id", step.agent_snapshot_files_artifact_id),
                ("agent_changelog_object_url", step.agent_changelog_object_url),
                ("enable_agent_changelog", step.enable_agent_changelog),
            ) if present
        )
    elif step.type in _AGENT_DEPLOYERS and agent in outside:
        why.append(f"{step.type} deploys {agent!r}, which is played from outside")
    elif step.type in _AGENT_TARGETING and agent in outside:
        why.append(f"{step.type} needs {agent!r} as a real agent, and it is played from outside")
    elif step.type == "collect_artifacts":
        if agent in outside and not (step.env_id or step.container_name or step.sandbox_name):
            why.append(f"collects from {agent!r}'s sandbox, and it is played from outside")
        if step.manifest_step_id in outside_prompts:
            why.append(f"its manifest is outside prompt {step.manifest_step_id!r}")
    elif step.type == "register_env_triggers" and step.executor_agent_name in outside:
        why.append(f"its executor {step.executor_agent_name!r} is played from outside")
    elif step.type == "peer_agents":
        why.extend(
            f"peers {name!r}, which is played from outside" for name in dict.fromkeys(
                name for peering in step.peerings for name in (peering.source_agent_name, *peering.peer_agent_names)
            ) if name in outside
        )
    elif step.type == "rubrics_verifier":
        if step.use_trajectory and step.prompt_id in outside_prompts.values():
            why.append("grades an outside prompt's trajectory, which an outside player doesn't hand back")
        if step.use_agent_judge and agent in outside:
            why.append(f"judges in {agent!r}'s sandbox, and it is played from outside")
    return why


def _top_level_verifiers(steps: list[TaskStep]) -> list[TaskStep]:
    aggregated = {vid for step in steps if step.type == "aggregate_verifiers" for vid in step.verifier_ids}
    return [step for step in steps if getattr(step, "verifier_id", None) and step.verifier_id not in aggregated]


def _agent_spec(steps: list[TaskStep], name: str) -> AgentSpec:
    deploy = next(step for step in steps if step.type == "deploy_agent" and step.agent_name == name)
    return AgentSpec(role=deploy.role, turns=tuple(
        TurnSpec(step_id=step.id, prompt=step.prompt, system_prompt=step.system_prompt or deploy.system_prompt)
        for step in steps if step.type == "prompt_agent" and step.agent_name == name
    ))


def _stand_in(step: TaskStep, deploys: dict[str, TaskStep]) -> TaskStep:
    """The step the episode's copy of the task runs in place of ``step``; ``deploys`` are the outside agents'."""
    if step.type == "deploy_agent" and step.agent_name in deploys:
        return _EpisodeAgentStep(
            id=step.id, version=step.version, depends_on=step.depends_on, fail_task_on_error=step.fail_task_on_error,
            agent_name=step.agent_name, env_ids=step.env_ids, env_step_id=step.env_step_id, role=step.role,
        )
    if step.type == "prompt_agent" and step.agent_name in deploys:
        timeout = step.timeout_seconds
        return _EpisodeTurnStep(
            id=step.id, version=step.version, depends_on=step.depends_on, fail_task_on_error=step.fail_task_on_error,
            agent_name=step.agent_name, prompt_id=step.prompt_id, prompt=step.prompt,
            system_prompt=step.system_prompt or deploys[step.agent_name].system_prompt,
            timeout_seconds=PromptAgentTaskStep.DEFAULT_PROMPT_TIMEOUT_SECONDS if timeout is None else timeout,
        )
    return step


class _EpisodeAgentStep(TaskStep):
    """An outside agent's ``deploy_agent``: deploys nothing, and binds the agent to its envs where it stood."""

    type: ClassVar[str] = "episode_agent"
    entity_refs = (EntityRef.env("env_ids[]"),)

    def __init__(
        self, id: str, version: Optional[int], *, agent_name: str, env_ids: list[str], env_step_id: str | None,
        role: str | None, depends_on: Optional[list] = None, fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.agent_name = agent_name
        self.env_ids = list(env_ids)
        self.env_step_id = env_step_id
        self.role = role

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "agent_name": self.agent_name, "env_ids": self.env_ids,
                "env_step_id": self.env_step_id, "role": self.role}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _EpisodeAgentStep:
        return cls(**cls._base_from_dict(data), agent_name=data["agent_name"], env_ids=data["env_ids"],
                   env_step_id=data["env_step_id"], role=data["role"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        raise NotImplementedError


class _EpisodeTurnStep(TaskStep):
    """An outside agent's ``prompt_agent``: hands the turn to the caller and records the reply it gets back."""

    type: ClassVar[str] = "episode_turn"
    entity_refs = ()

    def __init__(
        self, id: str, version: Optional[int], *, agent_name: str, prompt_id: str, prompt: str,
        system_prompt: str | None, timeout_seconds: int, depends_on: Optional[list] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.agent_name = agent_name
        self.prompt_id = prompt_id
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.timeout_seconds = timeout_seconds

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "agent_name": self.agent_name, "prompt_id": self.prompt_id,
                "prompt": self.prompt, "system_prompt": self.system_prompt, "timeout_seconds": self.timeout_seconds}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _EpisodeTurnStep:
        return cls(**cls._base_from_dict(data), agent_name=data["agent_name"], prompt_id=data["prompt_id"],
                   prompt=data["prompt"], system_prompt=data["system_prompt"],
                   timeout_seconds=data["timeout_seconds"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        raise NotImplementedError
