# Open txt data/datasets/goat_bench/hm3d/v1/val_unseen/lang_desc_episodes.txt
with open('data/datasets/goat_bench/hm3d/v1/val_unseen/lang_desc_episodes.txt', 'r') as f:
    desc_episodes = f.readlines()

    desc = []
    for line in desc_episodes:
        desc.append(line.split('|')[-1].strip())

# check unique descriptions
unique_desc = list(set(desc))

#save unique desc 
with open('data/datasets/goat_bench/hm3d/v1/val_unseen/unique_desc.txt', 'w') as f:
    for item in unique_desc:
        f.write("|  | " + item + '\n')

desc_llm_cat = []
for i, desc in enumerate(desc_episodes):
    for j, u_desc in enumerate(unique_desc):
        if u_desc == desc.line.split('|')[-1].strip():
            # add category to the description
            desc_episodes[i] = desc + '|' + desc_llm_cat[j]


