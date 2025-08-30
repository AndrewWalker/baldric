from baldric.commands.samples import sample_problem_configs
from baldric.commands.factory import create_problem


def test_create_examples():
    for name, problem_config in sample_problem_configs().items():
        p = create_problem(problem_config)
