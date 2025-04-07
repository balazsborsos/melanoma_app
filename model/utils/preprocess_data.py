import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil # Added for file operations

def split_data(csv_path, img_src_dir, target_dir, 
               diagnostic_col='diagnostic', image_id_col='image_id', file_ext='.jpg',
               val_size=0.1, test_size=0.1, random_state=42):
    """
    Reads a metadata CSV, performs a stratified split, adds an 'ml_set' column 
    to the DataFrame, and moves image files to corresponding train/val/test subdirectories.

    Args:
        csv_path (str): Path to the metadata CSV file.
        img_src_dir (str): Path to the directory containing the original image files.
        target_dir (str): Path to the directory where 'train', 'val', 'test' subdirectories 
                          will be created and images moved into. The modified CSV will implicitly
                          reference images in these new locations.
        diagnostic_col (str): Name of the column containing the class labels for stratification.
        image_id_col (str): Name of the column containing the unique image identifiers (without extension).
        file_ext (str): File extension of the images (e.g., '.jpg', '.png').
        val_size (float): Proportion of the dataset to include in the validation split (0.0 to 1.0).
        test_size (float): Proportion of the dataset to include in the test split (0.0 to 1.0).
        random_state (int, optional): Controls the shuffling applied to the data before splitting. 
                                      Pass an int for reproducible output across multiple function calls.

    Returns:
        pandas.DataFrame: The original DataFrame with an added 'ml_set' column ('train', 'val', 'test').
                         Returns None if the CSV file cannot be read or critical errors occur.

    Raises:
        ValueError: If val_size + test_size >= 1.0 or if sizes are negative.
        FileNotFoundError: If csv_path or img_src_dir do not exist.
        KeyError: If diagnostic_col is not in the CSV.
    """
    # --- Input Validation --- 
    if not (0 <= val_size < 1 and 0 <= test_size < 1 and (val_size + test_size) < 1):
        raise ValueError("val_size and test_size must be between 0 and 1, and their sum must be less than 1.")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Error: CSV file not found at {csv_path}")
    if not os.path.isdir(img_src_dir):
        raise FileNotFoundError(f"Error: Image source directory not found at {img_src_dir}")

    # --- Read Metadata --- 
    try:
        metadata_df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

    if diagnostic_col not in metadata_df.columns:
        raise KeyError(f"Error: Diagnostic column '{diagnostic_col}' not found in CSV.")
    if image_id_col not in metadata_df.columns:
        # Image ID is crucial for moving files
        raise KeyError(f"Error: Image ID column '{image_id_col}' not found in CSV.")

    # Drop rows with NaN in diagnostic column for stratification
    if metadata_df[diagnostic_col].isnull().any():
        print(f"Warning: NaN values found in the stratification column '{diagnostic_col}'. Rows with NaN will be dropped.")
        initial_count = len(metadata_df)
        metadata_df = metadata_df.dropna(subset=[diagnostic_col])
        print(f"Dropped {initial_count - len(metadata_df)} rows due to NaN in '{diagnostic_col}'.")
        if metadata_df.empty:
            print("Error: No data left after dropping NaN values in diagnostic column.")
            return None
            
    # Check for duplicate image IDs before proceeding
    if metadata_df[image_id_col].duplicated().any():
        print(f"Warning: Duplicate image IDs found in column '{image_id_col}'. Ensure IDs are unique for correct file moving.")
        # Depending on requirements, could raise error or just warn

    # --- Perform Stratified Splitting (to get indices) --- 
    
    # Initialize ml_set column
    metadata_df['ml_set'] = None
    df_to_split = metadata_df.copy()
    train_indices, val_indices, test_indices = [], [], []

    # Split off test set first
    if test_size > 0:
        remaining_indices = df_to_split.index
        try:
            train_val_indices, test_indices = train_test_split(
                remaining_indices,
                test_size=test_size,
                random_state=random_state,
                stratify=df_to_split.loc[remaining_indices, diagnostic_col]
            )
            metadata_df.loc[test_indices, 'ml_set'] = 'test'
            df_to_split = df_to_split.loc[train_val_indices] # Update df for next split
        except ValueError as e:
            print(f"Warning: Could not stratify test split (possibly too few samples in a class): {e}. Splitting without stratification.")
            train_val_indices, test_indices = train_test_split(
                remaining_indices,
                test_size=test_size,
                random_state=random_state
            )
            metadata_df.loc[test_indices, 'ml_set'] = 'test'
            df_to_split = df_to_split.loc[train_val_indices]
    else:
         train_val_indices = df_to_split.index # All remaining are train/val
         test_indices = pd.Index([]) # Empty index

    # Split off validation set from the remaining train_val data
    if val_size > 0 and not df_to_split.empty:
        # Adjust val_size relative to the remaining data
        relative_val_size = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0
        if relative_val_size >= 1.0: relative_val_size = 0.99 # Avoid splitting everything into val
        
        if relative_val_size > 0:
            try:
                train_indices, val_indices = train_test_split(
                    train_val_indices, # Use indices from previous step
                    test_size=relative_val_size,
                    random_state=random_state,
                    stratify=metadata_df.loc[train_val_indices, diagnostic_col] # Stratify based on original df slice
                )
                metadata_df.loc[val_indices, 'ml_set'] = 'val'
                metadata_df.loc[train_indices, 'ml_set'] = 'train'
            except ValueError as e:
                print(f"Warning: Could not stratify train/validation split: {e}. Splitting without stratification.")
                train_indices, val_indices = train_test_split(
                    train_val_indices,
                    test_size=relative_val_size,
                    random_state=random_state
                )
                metadata_df.loc[val_indices, 'ml_set'] = 'val'
                metadata_df.loc[train_indices, 'ml_set'] = 'train'
        else: # val_size was 0 or became 0 after test split adjustment
             metadata_df.loc[train_val_indices, 'ml_set'] = 'train' # All remaining are train
             train_indices = train_val_indices
             val_indices = pd.Index([])
    elif not df_to_split.empty: # No validation split needed, all remaining are train
        metadata_df.loc[train_val_indices, 'ml_set'] = 'train'
        train_indices = train_val_indices
        val_indices = pd.Index([])
    else: # df_to_split was empty (e.g. test_size was ~1)
        train_indices = pd.Index([])
        val_indices = pd.Index([])

    print("Split assignment complete:")
    print(metadata_df['ml_set'].value_counts())    
    
    # --- Create Target Directories --- 
    train_path = os.path.join(target_dir, 'train')
    val_path = os.path.join(target_dir, 'val')
    test_path = os.path.join(target_dir, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    print(f"Created target directories: {train_path}, {val_path}, {test_path}")

    # --- Move Image Files --- 
    moved_count = 0
    skipped_count = 0
    error_count = 0
    print(f"Moving images from '{img_src_dir}' to '{target_dir}' subdirectories...")

    # Use itertuples for efficiency
    for row in metadata_df.itertuples(index=False):
        image_id = getattr(row, image_id_col)
        ml_set = row.ml_set
        
        if pd.isna(image_id) or not ml_set:
            print(f"Skipping row due to missing image_id or ml_set assignment: {row}")
            skipped_count += 1
            continue
            
        src_img_filename = f"{str(image_id)}{file_ext}"
        src_img_path = os.path.join(img_src_dir, src_img_filename)

        target_subdir = os.path.join(target_dir, ml_set)
        target_img_path = os.path.join(target_subdir, src_img_filename)

        if os.path.exists(src_img_path):
            try:
                shutil.move(src_img_path, target_img_path)
                moved_count += 1
            except OSError as e:
                print(f"Error moving file {src_img_path} to {target_img_path}: {e}")
                error_count += 1
            except Exception as e:
                 print(f"Unexpected error moving file {src_img_path}: {e}")
                 error_count += 1
        else:
            print(f"Warning: Source image file not found, skipped: {src_img_path}")
            skipped_count += 1

    print(f"Image moving complete: {moved_count} moved, {skipped_count} skipped, {error_count} errors.")

    return metadata_df

# Example usage:
if __name__ == '__main__':
    # --- Create dummy data for demonstration --- 
    print("Creating dummy data for demonstration...")
    base_dummy_dir = 'dummy_data_preprocess_move'
    dummy_csv_path = os.path.join(base_dummy_dir, 'dummy_metadata.csv')
    dummy_img_src_dir = os.path.join(base_dummy_dir, 'images_source')
    dummy_target_dir = os.path.join(base_dummy_dir, 'images_split')
    
    # Clean up previous run if necessary
    if os.path.exists(base_dummy_dir):
        print(f"Cleaning up previous dummy data in '{base_dummy_dir}'...")
        shutil.rmtree(base_dummy_dir)
        
    os.makedirs(dummy_img_src_dir, exist_ok=True)
    os.makedirs(dummy_target_dir, exist_ok=True)

    num_samples = 200
    data = {
        'image_id': [f'img_{i:03d}' for i in range(num_samples)],
        'patient_id': [f'patient_{i % 50:03d}' for i in range(num_samples)], 
        'lesion_id': [f'lesion_{i:03d}' for i in range(num_samples)],
        'diagnostic': [
            'ClassA' if i < num_samples * 0.6 else 
            'ClassB' if i < num_samples * 0.9 else 
            'ClassC' for i in range(num_samples)
            ],
        'feature1': [i * 0.1 for i in range(num_samples)]
    }
    dummy_df = pd.DataFrame(data)
    # Add a row with NaN diagnostic to test dropping
    # dummy_df.loc[num_samples] = {'image_id': f'img_{num_samples:03d}', 'diagnostic': None, 'patient_id': 'patient_999', 'lesion_id': 'lesion_999', 'feature1': 0.0 }
    # Add a row with missing image_id to test skipping
    # dummy_df.loc[num_samples+1] = {'image_id': None, 'diagnostic': 'ClassA', 'patient_id': 'patient_998', 'lesion_id': 'lesion_998', 'feature1': 0.1 }
    
    dummy_df.to_csv(dummy_csv_path, index=False)
    print(f"Dummy metadata saved to '{dummy_csv_path}'")
    
    # Create dummy image files
    print(f"Creating {num_samples} dummy image files in '{dummy_img_src_dir}'...")
    for img_id in dummy_df[dummy_df['image_id'].notna()]['image_id']:
        img_filename = f"{img_id}.jpg"
        img_path = os.path.join(dummy_img_src_dir, img_filename)
        with open(img_path, 'w') as f:
            f.write(f"content_of_{img_id}") # Create empty/dummy file
    print("Dummy image files created.")
    
    print("Original class distribution:")
    print(dummy_df['diagnostic'].value_counts(normalize=True))
    print("-"*30)
    # --- End of dummy data creation ---

    try:
        print(f"Attempting to split data from '{dummy_csv_path}' and move images...")
        modified_metadata = split_data(
            csv_path=dummy_csv_path, 
            img_src_dir=dummy_img_src_dir,
            target_dir=dummy_target_dir,
            diagnostic_col='diagnostic',
            image_id_col='image_id', 
            file_ext='.jpg',
            val_size=0.15, 
            test_size=0.15, 
            random_state=123
        )

        if modified_metadata is not None:
            print("\n--- Split Results --- ")
            print("Distribution in 'ml_set' column:")
            print(modified_metadata['ml_set'].value_counts())
            print("\nClass distribution within each set:")
            print(modified_metadata.groupby('ml_set')['diagnostic'].value_counts(normalize=True))
            print(f"\nModified metadata head:\n{modified_metadata.head()}")
            print(f"\nImages have been moved into subfolders within: '{dummy_target_dir}'")
            
            # --- Optional: Save the modified DataFrame --- 
            modified_csv_path = os.path.join(dummy_target_dir, 'metadata_split.csv')
            modified_metadata.to_csv(modified_csv_path, index=False)
            print(f"\nModified metadata saved to '{modified_csv_path}'")

    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"\nError during data splitting/moving: {e}")
    finally:
        # Optional: Clean up dummy data automatically
        # if os.path.exists(base_dummy_dir):
        #     print(f"\nCleaning up dummy data directory '{base_dummy_dir}'...")
        #     shutil.rmtree(base_dummy_dir)
        print(f"\nDummy data location: {base_dummy_dir}")
        pass # Keep dummy data for inspection 