import os

def load_data_from_shard(shard_dir):
    """
    Load image paths and captions from a shard directory, filtering by agreement.txt.

    Args:
        shard_dir (str): Path to the shard directory (e.g., "data/spatial_twoshapes/agreement/relational/test/shard0").

    Returns:
        tuple: A tuple containing:
            - image_paths (list of str): List of valid image paths.
            - captions (list of str): List of valid captions corresponding to the images.
    """
    # Define file paths
    caption_file = os.path.join(shard_dir, "caption.txt")
    agreement_file = os.path.join(shard_dir, "agreement.txt")

    # Read captions and agreement values
    with open(caption_file, "r") as f:
        captions = f.readlines()
    with open(agreement_file, "r") as f:
        agreements = f.readlines()

    # Ensure the lengths match
    assert len(captions) == len(agreements), "Mismatch between captions and agreement entries."

    # Filter valid captions and corresponding image paths
    image_paths = []
    valid_captions = []
    for i, (caption, agreement) in enumerate(zip(captions, agreements)):
        if float(agreement.strip()) == 1.0:  # Only include valid captions
            image_name = f"world-{i}.png"
            image_path = os.path.join(shard_dir, image_name)
            if os.path.exists(image_path):  # Ensure the image file exists
                image_paths.append(image_path)
                valid_captions.append(caption.strip())

    return image_paths, valid_captions


def load_all_shards(base_dir, shard_count=5):
    """
    Load data from all shard directories.

    Args:
        base_dir (str): Base directory containing shard directories.
        shard_count (int): Number of shard directories to process.

    Returns:
        tuple: A tuple containing:
            - all_image_paths (list of str): List of all valid image paths.
            - all_captions (list of str): List of all valid captions.
    """
    all_image_paths = []
    all_captions = []

    for shard_num in range(shard_count):
        shard_dir = os.path.join(base_dir, f"shard{shard_num}")
        if os.path.exists(shard_dir):
            image_paths, captions = load_data_from_shard(shard_dir)
            all_image_paths.extend(image_paths)
            all_captions.extend(captions)

    return all_image_paths, all_captions


# Example usage
if __name__ == "__main__":
    base_dir = "/home/maqima/VLM-Visualizer/data/spatial_twoshapes/agreement/relational/test"
    image_paths, captions = load_all_shards(base_dir)

    # Print the results
    print("Number of valid image-caption pairs:", len(image_paths))
    for img, cap in zip(image_paths[:5], captions[:5]):  # Print first 5 pairs as a sample
        print(f"Image: {img}, Caption: {cap}")