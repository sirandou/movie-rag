# ML Component Instructions

## Framework Preferences
- PyTorch Lightning over raw PyTorch
- scikit-learn for classical ML
- pandas for data manipulation

## ML Best Practices
- Include random seeds for reproducibility
- Clear train/validation/test splits
- Use wandb for experiment tracking
- Type hint tensor shapes when possible
- Separate data processing from model code

## Common Patterns
```python
# Preferred model structure
class MyModel(pl.LightningModule):
    def __init__(self, config: ModelConfig):
        # Use config objects for hyperparameters
```

## Testing
- Mock data loaders in tests
- Test preprocessing functions separately
- Include data validation checks