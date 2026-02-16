.PHONY: format help

help:
	@echo "Available targets:"
	@echo "  format    - Format code with black and isort"

format:
	@echo "Formatting code with isort and black..."
	isort src/
	black src/
	@echo "Formatting complete!"
