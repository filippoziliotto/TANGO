from habitat_baselines.rl.ppo.utils.utils import PromptUtils
"""
Prompt examples and utils for Objectnav task
"""

def generate_multinav_prompt(prompt_utils: PromptUtils):
    object_cats = prompt_utils.get_multinav_target()

    # if object_cat has space change them with "_"
    print("Navigating to object:", object_cats)
    object_cats_new = []
    for cat_var in object_cats:
        # names is "cylinder_{color}"
        # transform to "Color Cylinder"

        cat = cat_var.split("_")
        cat = cat[1] + " " + cat[0]
        object_cats_new.append((cat_var,cat))

    prompt = f"""     
explore_scene()
{object_cats_new[0][0]} = detect('{object_cats_new[0][1]}')
if is_found({object_cats_new[0][0]}):
    navigate_to({object_cats_new[0][0]})
    return
    explore_scene()
    {object_cats_new[1][0]} = detect('{object_cats_new[1][1]}')
    if is_found({object_cats_new[1][0]}):
        navigate_to({object_cats_new[1][0]})
        return
        explore_scene()
        {object_cats_new[2][0]} = detect('{object_cats_new[2][1]}')
        if is_found({object_cats_new[2][0]}):
            navigate_to({object_cats_new[2][0]})
            return
"""
    return prompt     
