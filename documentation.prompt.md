# How create documentation with AI prompt

## 1. Setting Up Documentation Templates

### Template Structure
```markdown
# [Component/Feature Name]

## Overview
[1-2 sentence description]

## API Reference
| Property | Type | Default | Description |
|----------|------|---------|-------------|
| prop1    | type | default | description |

## Usage Examples
```jsx
// Basic usage example
```

## Common Patterns
[Common implementation patterns]

## Troubleshooting
[FAQs and solutions]
```

### Implementation Steps:
1. Create template files in a `/docs/templates/` directory
2. Add a pre-commit hook that validates new docs against templates
3. Include "template markers" that AI can recognize:
   ```markdown
   <!-- TEMPLATE:API_REFERENCE -->
   <!-- Content here -->
   <!-- END_TEMPLATE -->
   ```
4. Create a template selection command in your editor/IDE

## 2. Chain of Thought Comments

### Effective Comment Structure
```javascript
/**
 * CONTEXT: This component handles form validation for the user registration flow
 * DECISION: Using uncontrolled form to improve performance with many fields
 * ALTERNATIVE CONSIDERED: Formik was evaluated but added unnecessary complexity
 * RELATED_FILES: /src/validators/user.js, /src/hooks/useFormSubmit.js
 */
```

### Practical Implementation:
1. Standardize comment prefixes: CONTEXT, DECISION, ALTERNATIVE, RELATED
2. Add special AI-targeted comments:
   ```javascript
   // AI-CONTEXT: This pattern is repeated throughout auth flows
   ```
3. Include reasoning chains in complex functions:
   ```javascript
   function validateUserInput(input) {
     // Step 1: Sanitize inputs to prevent XSS
     // Step 2: Validate email format
     // Step 3: Check password requirements
   }
   ```
4. Document edge cases explicitly:
   ```javascript
   // EDGE CASE: API returns null for inactive users
   ```

## 3. Versioning Documentation with Code

### Directory Structure
```
/src
  /components
    /Button
      Button.jsx
      Button.test.js
      Button.docs.md  <-- Documentation lives with component
/docs
  /compiled          <-- Built documentation site
  /api               <-- Auto-generated API docs
  /guides            <-- Conceptual documentation
    /v1              <-- Historical versions
    /v2              <-- Current version
```

### Version Control Strategies:
1. **Branch Documentation with Features**:
   - Create feature branches that include both code and docs
   - Use PR templates that require documentation updates

2. **Documentation Versioning System**:
   ```javascript
   /**
   * @since v2.3.0
   * @deprecated v3.1.0 Use NewComponent instead
   */
   ```

3. **Automated Documentation Testing**:
   - Set up tests that verify code examples in documentation actually compile
   - Create GitHub Actions workflow:
   ```yaml
   name: Docs Code Validation
   on: [push, pull_request]
   jobs:
     test-doc-examples:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - name: Extract and test code samples
           run: ./scripts/test-doc-examples.sh
   ```

4. **Documentation Change Tracking**:
   - Add `CHANGELOG.md` specifically for documentation
   - Use tagging system for documentation versions
