
import json
import os
from pathlib import Path

def check_data():
    questions_file = r"d:/cnn/data/raw/vqa_v2/questions/v2_OpenEnded_mscoco_val2014_questions.json"
    images_dir = r"d:/cnn/data/raw/coco_val2017/images"
    
    print(f"Checking data alignment:")
    print(f"Questions: {questions_file}")
    print(f"Images: {images_dir}")
    
    if not os.path.exists(questions_file):
        print("ERROR: Questions file not found!")
        return
        
    if not os.path.exists(images_dir):
        print("ERROR: Images directory not found!")
        return

    # Load questions
    print("Loading questions...")
    with open(questions_file, 'r') as f:
        data = json.load(f)
        questions = data['questions']
        
    print(f"Total questions in VQA val2014: {len(questions)}")
    
    # Get all image IDs from questions
    q_image_ids = set(q['image_id'] for q in questions)
    print(f"Unique image IDs in questions: {len(q_image_ids)}")
    
    # Get local image files
    print("Scanning image directory...")
    image_files = os.listdir(images_dir)
    print(f"Total files in image dir: {len(image_files)}")
    
    # Parse image IDs from filenames (format: 000000123456.jpg)
    local_image_ids = set()
    for fname in image_files:
        if fname.endswith('.jpg'):
            try:
                # remove .jpg and parse int
                image_id = int(fname.split('.')[0])
                local_image_ids.add(image_id)
            except ValueError:
                pass
                
    print(f"Unique image IDs in local dir: {len(local_image_ids)}")
    
    # Check intersection
    intersection = q_image_ids.intersection(local_image_ids)
    print(f"\nINTERSECTION: {len(intersection)} images found in both.")
    
    if len(intersection) == 0:
        print("CRITICAL: No matching images found. Dataset loader will yield 0 samples.")
        print("Reason: Image filenames/IDs do not match.")
    else:
        # Calculate how many questions this covers
        valid_questions = [q for q in questions if q['image_id'] in intersection]
        print(f"Valid questions (samples) available: {len(valid_questions)}")
        print(f"Percentage of original VQA: {len(valid_questions)/len(questions)*100:.2f}%")

if __name__ == "__main__":
    check_data()
