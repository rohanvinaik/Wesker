# Security policy

## Reporting a vulnerability

Report security issues privately: open this repository's **Security** tab and choose **Report a
vulnerability** (<https://github.com/rohanvinaik/Wesker/security/advisories/new>). Please don't open a
public issue for anything you believe is exploitable.

A useful report says what you ran, what happened, and what you expected; a minimal reproduction is best.
Wesker is maintained by one person. Reports are answered as quickly as is realistic, and there is no bug
bounty.

## Supported versions

Fixes land on `main` and ship in the next release of `Wesker`. Only the latest release is supported.

## What Wesker executes, by design

Wesker is a mutation engine, so it runs the code you point it at. It imports the target module, compiles
and executes mutants of its functions, and runs the repository's test suite against them, either
in-process or in isolated worker processes that it starts and kills. Running Wesker on a repository needs
the same trust as running its tests.

The GitHub Action installs the project under test and runs its suite. Its `install` input is executed as a
shell command by design, so it must come from the workflow author, never from untrusted text such as a
pull request title. Every other input reaches Wesker as data through the environment, not spliced into the
script.

What it writes: caches and reports under `.wesker/`, and the SARIF file the Action is asked for. The
package imports no network client.

In scope is anything beyond that: executing code outside the repository it was pointed at, writing
outside those places, or sending data off the machine.
