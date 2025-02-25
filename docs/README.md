# Ragatanga Documentation

This directory contains the documentation for Ragatanga, built with MkDocs and the Material theme.

## Documentation Structure

- `index.md`: Home page and project overview
- `getting-started.md`: Installation and basic setup
- `usage.md`: Detailed usage examples
- `configuration.md`: Configuration options
- `architecture.md`: Technical architecture overview
- `api-reference.md`: API documentation
- `contributing.md`: Contributing guidelines
- `changelog.md`: Version history

## Building the Documentation

To build the documentation:

```bash
# Install dependencies
pip install mkdocs mkdocs-material mkdocstrings mkdocstrings-python

# Build the documentation
mkdocs build

# Serve the documentation locally
mkdocs serve
```

## Deployment

To deploy the documentation to GitHub Pages:

```bash
mkdocs gh-deploy
```

## Customization

The documentation is configured in `mkdocs.yml` in the project root. You can customize:

- Theme and appearance
- Navigation structure
- Plugins and extensions
- Metadata

## Adding API Documentation

As the codebase develops, you can add automatic API documentation using mkdocstrings:

1. Ensure your Python code has proper docstrings
2. Update the `api-reference.md` file to use the `mkdocstrings` plugin:

```markdown
# API Reference

## OntologyManager

::: ragatanga.core.ontology.OntologyManager
    options:
      show_root_heading: true
      show_source: true
```

## Next Steps

- Add more detailed examples
- Include diagrams and visualizations
- Add a FAQ section
- Improve API documentation with actual code references
- Add a search index 