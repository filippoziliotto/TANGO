import os
import pickle
import json
from habitat.utils.geometry_utils import quaternion_to_list

def load_qa_data(qa_json_path):
    """Load the questions and answers data."""
    with open(qa_json_path, 'r') as f:
        qa_data = json.load(f)

    # Create a mapping from episode_history to a list of questions and answers
    qa_mapping = {}
    for entry in qa_data:
        episode_history = entry["episode_history"]
        if episode_history.startswith("hm3d-v0"):
            if episode_history not in qa_mapping:
                qa_mapping[episode_history] = []
            qa_mapping[episode_history].append({
                "question_text": entry["question"],
                "answer_text": entry["answer"]
            })
    return qa_mapping

def load_pickle_data(pickle_path):
    """Load data from a pickle file."""
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def create_episode(episode_id, scene_name, first_data, last_data, qa):
    """Create an episode dictionary."""
    scene_name_code = scene_name.split("-")[-1]
    return {
        "episode_id": str(episode_id),
        "scene_id": f"data/scene_datasets/hm3d_v0.2/val/{scene_name}/{scene_name_code}.basis.glb",
        "scene_dataset_config": "./data/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json",
        # array to list the next line
        "start_position": first_data["agent_state"].position.tolist(),
        "start_rotation": quaternion_to_list(first_data["agent_state"].rotation.tolist()),
        "info": {},
        "goals": [
            {
                "position": last_data["agent_state"].position.tolist(),
                "radius": None
            }
        ],
        "question": qa,
        "shortest_paths": None,
    }

def get_scene_name_mapping(scene_dir):
    """Get a mapping from original scene names to formatted scene names."""
    scene_mapping = {}
    for scene in os.listdir(scene_dir):
        if os.path.isdir(os.path.join(scene_dir, scene)):
            scene_id = scene.split("-")[-1]
            scene_mapping[scene_id] = scene
    return scene_mapping

def process_subfolders(base_dir, qa_mapping, scene_mapping):
    """Process each subfolder to generate episodes."""
    episodes_data = {"episodes": []}
    episode_id = 0

    for subfolder in sorted(os.listdir(base_dir)):
        subfolder_path = os.path.join(base_dir, subfolder)
        
        if os.path.isdir(subfolder_path):
            pkl_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith('.pkl')])
            
            if not pkl_files:
                continue
            
            first_data = load_pickle_data(os.path.join(subfolder_path, pkl_files[0]))
            last_data = load_pickle_data(os.path.join(subfolder_path, pkl_files[-1]))
            
            episode_history_key = f"hm3d-v0/{subfolder}"
            scene_id = subfolder.split("-")[-1]
            
            if episode_history_key in qa_mapping and scene_id in scene_mapping:
                scene_name = scene_mapping[scene_id]
                
                for qa in qa_mapping[episode_history_key]:
                    episode = create_episode(episode_id, scene_name, first_data, last_data, qa)
                    episodes_data["episodes"].append(episode)
                    episode_id += 1

    return episodes_data

def save_episodes_data(output_json_path, episodes_data):
    """Save the episodes data to a JSON file."""
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as json_file:
        json.dump(episodes_data, json_file, indent=4)

def main():
    base_dir = "/home/ziliottf/repos/open-eqa/data/frames/hm3d-v0"
    qa_json_path = "/home/ziliottf/repos/open-eqa/data/open-eqa-v0.json"
    scene_dir = "/home/ziliottf/repos/navprog/data/scene_datasets/hm3d_v0.2/val/"
    output_json_path = "/home/ziliottf/repos/navprog/data/datasets/open_eqa/val/val.json"

    qa_mapping = load_qa_data(qa_json_path)
    scene_mapping = get_scene_name_mapping(scene_dir)
    episodes_data = process_subfolders(base_dir, qa_mapping, scene_mapping)
    save_episodes_data(output_json_path, episodes_data)
    print(f"JSON file created at {output_json_path}")

if __name__ == "__main__":
    main()
