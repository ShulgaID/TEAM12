import nox


@nox.session(python=["3.10", "3.11", "3.12"])
def tests(session):
    session.install(".[dev]")
    session.run("pytest", "tests/", "-v")


@nox.session
def lint(session):
    session.install("ruff")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")
