from .ddim_flow_scheduler import DDIMFlowScheduler
from .flow_match_euler_discrete_scheduler import FlowMatchEulerDiscreteScheduler
from .linear_scheduler import LinearScheduler
from .stork_scheduler import STORKScheduler

__all__ = [
    "LinearScheduler",
    "FlowMatchEulerDiscreteScheduler",
    "DDIMFlowScheduler",
    "STORKScheduler",
]


class SchedulerModuleNotFound(ValueError): ...


class SchedulerClassNotFound(ValueError): ...


class InvalidSchedulerType(TypeError): ...


# Factory functions for STORK variants
def _create_stork_2(runtime_config, **kwargs):
    """Create STORK-2 (second-order) scheduler."""
    kwargs.setdefault("order", 2)
    return STORKScheduler(runtime_config, **kwargs)


def _create_stork_4(runtime_config, **kwargs):
    """Create STORK-4 (fourth-order) scheduler."""
    kwargs.setdefault("order", 4)
    return STORKScheduler(runtime_config, **kwargs)


SCHEDULER_REGISTRY = {
    "linear": LinearScheduler,
    "LinearScheduler": LinearScheduler,
    "flow_match_euler_discrete": FlowMatchEulerDiscreteScheduler,
    "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
    "ddim": DDIMFlowScheduler,  # DDIM-style Flow Matching
    "DDIMFlowScheduler": DDIMFlowScheduler,
    "stork": _create_stork_2,  # STORK: Stabilized Runge-Kutta (default to order 2)
    "stork-2": _create_stork_2,  # STORK-2 (second-order, faster)
    "stork-4": _create_stork_4,  # STORK-4 (fourth-order, more accurate)
    "STORKScheduler": STORKScheduler,
}


def register_contrib(scheduler_object, scheduler_name=None):
    if scheduler_name is None:
        scheduler_name = scheduler_object.__name__
    SCHEDULER_REGISTRY[scheduler_name] = scheduler_object


def try_import_external_scheduler(scheduler_object_path: str):
    import importlib
    import inspect

    from .base_scheduler import BaseScheduler

    try:
        last_dot_index = scheduler_object_path.rfind(".")

        if last_dot_index < 0:
            raise SchedulerModuleNotFound(
                f"Invalid scheduler path format: {scheduler_object_path!r}. "
                "Expected format: some_library.some_package.maybe_sub_package.YourScheduler"
            )

        module_name_str = scheduler_object_path[:last_dot_index]
        scheduler_class_name = scheduler_object_path[last_dot_index + 1 :]
        module = importlib.import_module(module_name_str)
    except ImportError:
        raise SchedulerModuleNotFound(scheduler_object_path)

    try:
        # Step 2: Get the object from the module using its string name
        SchedulerClass = getattr(module, scheduler_class_name)
    except AttributeError:
        raise SchedulerClassNotFound(scheduler_object_path)

    # Step 3: Validate that it's a class and inherits from BaseScheduler
    if not inspect.isclass(SchedulerClass):
        raise InvalidSchedulerType(
            f"{scheduler_object_path!r} is not a class. Schedulers must be classes inheriting from BaseScheduler."
        )

    if not issubclass(SchedulerClass, BaseScheduler):
        raise InvalidSchedulerType(
            f"{scheduler_object_path!r} does not inherit from BaseScheduler. "
            f"All schedulers must inherit from mflux.schedulers.BaseScheduler."
        )

    return SchedulerClass
