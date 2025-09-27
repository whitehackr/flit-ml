# Claude Code Assistant Configuration

## Documentation Standards

### PR Comment Technical Depth Requirements

All PR comments documenting technical decisions must meet **Sr Principal Engineer** standards:

#### Content Depth
- **Educational First**: Write to teach junior, mid, and senior engineers simultaneously
- **Technical Rigor**: Explain not just what was built, but why design decisions were made
- **Trade-off Analysis**: Document alternatives considered and rejection rationale
- **Context Setting**: Establish problem space before presenting solutions
- **Probabilistic Thinking**: When applicable, explain statistical/ML reasoning in depth

#### Writing Style
- **Prose Over Bullets**: Use solid prose for complex explanations; bullets only for simple lists
- **Human Voice**: Sound like an experienced engineer teaching, not AI-generated content
- **No Attribution**: Never reference "Sr Principal" titles, experience levels, or self-promotion. Also, no attributions to Claude Code.
- **Depth Without Repetition**: Be comprehensive but never redundant

#### Technical Documentation Pattern
```markdown
## Problem Context
[Establish the technical challenge and why it matters]

### Design Decision: [Clear title]
[Explain the chosen approach with code examples]

### Alternative Approaches Considered
[Detail what was rejected and why]

### Technical Deep Dive
[Explain the underlying principles, statistics, algorithms]

### Trade-offs & Future Considerations
[Long-term implications and improvement paths]
```

#### Code Quality Standards
- Always show working code examples
- Include performance metrics and benchmarks
- Document edge cases and error handling
- Explain statistical/mathematical foundations when relevant

### Development Commands

Standard commands for this repository:

#### Testing
```bash
# Run all tests
poetry run pytest

# Test specific components
poetry run pytest tests/integration/
poetry run python test_single_transaction.py
poetry run python test_predictor.py
```

#### Linting & Type Checking
```bash
# Run before committing
poetry run ruff check .
poetry run mypy .
```

#### Model Development
```bash
# Activate environment
poetry shell

# Production artifact validation
poetry run pytest tests/integration/ -v
```

### Commit Message Standards

- **Concise but complete**: Summarize the change and key technical points
- **No excessive detail**: Technical depth belongs in PR comments, not commit messages
- **Imperative mood**: "Implement X" not "Implemented X"
- **Reference documentation**: When complex decisions are made, reference PR comments

### Known Issues Management

Maintain `docs/models/domain_monthyr_known_issues.md` with:
- **Clear prioritization**: 🔴 Blocking, 🟡 Important, 🟢 Enhancement
- **Technical solutions**: Specific code examples for fixes
- **Business impact**: How issues affect production deployment
- **Timeline guidance**: When to address each issue

### Educational Philosophy

Every technical document should enable a junior/mid/sr engineer to:
1. **Understand the problem** being solved
2. **Learn the reasoning** behind design decisions
3. **Implement similar solutions** in different contexts
4. **Recognize trade-offs** in their own work

This documentation serves as both project record and engineering education resource.