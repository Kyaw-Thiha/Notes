## Testing Levels
`Unit Testing`: Testing methods individually
`Module Testing`: Testing self-related module of codes
`Integration Testing`: Testing how modules interact
`System Testing`: Testing overall functionality of system
`Acceptance Testing`: Testing whether software is acceptable to user

## Testing Approaches
- `Black-Box Approach`: Derive tests form external descriptions (related to `Test-Drive Development`)
- `White-Box Approach`: Derive tests from code internals of software

## Criterion-Based Test Design
Define a `Coverage Criterion`; rules on test requirements (like every function must have 1 test case)

`Limitations`
- Dead code in codebase cannot be tested
- Complex criteria make it impossible to have full coverage
- Full coverage does note implies no bugs (maybe edge cases or different hardware)

## Different Errors
`Software Fault`: Static defect in software
`Software Error`: Incorrect internal state that is manifestation of some fault
`Software Failure`: External, incorrect behavior with respect to requirements

## RIPR Model
Four conditions must fulfill for an error to be observed.
- `Reachability`: Test must reach the error
- `Infection`: After faulty location is executed, state of program must be incorrect
- `Propagation`: Infected state must propagate to rest of execution, and cause final state of program to be incorrect
- `Revealability`: Tester must observe part of incorrect portion of final program state (cannot do for `seg faults`)
