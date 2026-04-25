# Software Development Cycle

## Waterfall
This is a traditional model that focus on a sequential model.
Its main weaknesses are
- Not okie when requirements change frequently
- Time gets squeezed further into the project

## Agile
- Agile & interactions > Processes & Tools
- Working software > Comprehensive Documentation
- Customer Collaboration > Contract Negotiation
- Responding to change > Following plan

`Examples`
- `Scrum`
- `Extreme Programming (XP)`
- `Test-Driven Development (TDD)`
- `Feature-Driven Development (FDD)`

## Scrum
The most used implementation of `Agile`
```
Product Backlog --> Scrum Backlog --> Sprint [1 - 4 weeks]
                                      [Stand-Up every 24hours]
```

Tasks in `Scrum` are broken into `User Stories`

- At the start of each `Sprint`, team agrees on taking specific number of `user stories`.
- Then, these tasks are done during the `Sprint`.
- After each of the `Sprint`, the team does `Retrospective`, where past sprint is discussed, and chances for improvements are raised.

## Scrum Terminologies

`User Story`
```python
As a {ACTOR/OBJECT} I want to {ACTION} so that {RESULT}
```

`Planning Session`
This is the meeting at the start of the sprint, and can take a few hours deciding on how much tasks to take on + any challenges.

`Stand-Up Meeting`
This is a daily meeting where
- Progress from last meeting
- Expected progress before next meeting
- Any impediments facing

`Definition of Done`
We can define done based on different criteria such as
- Fully unit tested
- Successfully integrated in rest of the code
- Peer Reviewed
- Fully Commented
