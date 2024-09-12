import numpy as np

def variance_between(values, classes):
    """
    Calculate the variance between classes.
    
    Args:
        values (array-like): A list or array of values.
        classes (array-like): A list or array of corresponding class labels.
        
    Returns:
        float: Variance between classes.
    """
    overall_mean = np.mean(values)
    unique_classes = np.unique(classes)
    
    between_var = 0
    for cls in unique_classes:
        class_values = values[classes == cls]
        class_mean = np.mean(class_values)
        n_cls = len(class_values)
        between_var += n_cls * (class_mean - overall_mean) ** 2
    
    return between_var / len(values)


def variance_within(values, classes):
    """
    Calculate the variance within classes.
    
    Args:
        values (array-like): A list or array of values.
        classes (array-like): A list or array of corresponding class labels.
        
    Returns:
        float: Variance within classes.
    """
    unique_classes = np.unique(classes)
    
    within_var = 0
    for cls in unique_classes:
        class_values = values[classes == cls]
        class_var = np.var(class_values, ddof=1)
        within_var += len(class_values) * class_var
    
    return within_var / len(values)

'''
# Example usage
values = np.array([5.1, 4.9, 4.7, 6.0, 5.8, 5.6])
classes = np.array([0, 0, 0, 1, 1, 1])

var_between = variance_between(values, classes)
var_within = variance_within(values, classes)

print(f"Variance Between Classes: {var_between}")
print(f"Variance Within Classes: {var_within}")
'''
