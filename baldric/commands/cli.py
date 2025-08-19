import click
import yaml
from loguru import logger
from .factory import load_problem, create_problem
from .samples import sample_problem_configs
from baldric.plotting import plot_problem, plot_problem_anim, plot_without_solve


@click.command()
@click.option("--problem", "-p", type=str, required=True)
@click.option("--output", "-o", type=str, default="out.png")
def solve(problem: str, output: str):
    p = load_problem(problem)
    plot_problem(p, output)


@click.command()
def create_samples():
    for name, problem_config in sample_problem_configs().items():
        filename = name + ".yml"
        logger.info(f"writing {filename}")
        with open(filename, "w") as fh:
            yaml.dump(problem_config.model_dump(), fh)


@click.command()
@click.option("--name", "-n", type=str, required=True)
@click.option("--anim", "-a", is_flag=True)
@click.option("--dryrun", "-d", is_flag=True, help="plot without solve")
def solve_sample(name: str, anim: bool, dryrun: bool):
    problem_config = sample_problem_configs()[name]
    p = create_problem(problem_config)
    if dryrun:
        plot_without_solve(p, name + ".png")
    elif anim:
        plot_problem_anim(p, name + ".gif")
    else:
        plot_problem(p, name + ".png")
    print(p.collision_checker.checks)


@click.command()
def solve_all_samples():
    for name, problem_config in sample_problem_configs().items():
        logger.info("solving", name)
        p = create_problem(problem_config)
        plot_problem(p, name + ".png")
