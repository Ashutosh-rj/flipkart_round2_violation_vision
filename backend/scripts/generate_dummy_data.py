import os
import cv2
import numpy as np

def generate_dataset(base_dir="dataset", samples_per_class=50):
    classes = ["0_no_helmet", "1_helmet"]
    splits = {"train": int(samples_per_class * 0.8), "val": int(samples_per_class * 0.2)}

    for split, count in splits.items():
        for cls in classes:
            dir_path = os.path.join(base_dir, split, cls)
            os.makedirs(dir_path, exist_ok=True)
            
            for i in range(count):
                # Create a random noise image (dummy)
                # To help the model actually learn something instead of random noise, 
                # we'll color them differently
                img = np.zeros((224, 224, 3), dtype=np.uint8)
                if "1_helmet" in cls:
                    img[:] = (200, 100, 100) # Blueish for helmet
                    cv2.circle(img, (112, 112), 50, (0, 0, 255), -1) # Red circle
                else:
                    img[:] = (100, 200, 100) # Greenish for no helmet
                    cv2.rectangle(img, (50, 50), (174, 174), (0, 255, 0), -1)
                
                # Add noise
                noise = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
                img = cv2.add(img, noise)
                
                filepath = os.path.join(dir_path, f"dummy_{i}.jpg")
                cv2.imwrite(filepath, img)

    print(f"Generated dummy dataset in {base_dir}")

if __name__ == "__main__":
    generate_dataset()
