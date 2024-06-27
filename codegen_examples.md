# In-Context Examples

## Object recognition
**Question**: What is the white object on the wall above the TV?  \
**Program**: 
```python
while not episode_is_over():
    explore_scene()
    object = detect_object('TV')
    if object:
        navigate_to(object)
        look_around()
        answer('What is the white object on the wall above the TV?')
        episode_is_over = True
```
## Attribute recognition
**Question**: What material is the floor? \
**Program**:
```python   
while not episode_is_over():
    look_around()
    image = segment_scene('floor')
    answer(image, 'What material is the floor?')
    episode_is_over = True
```

## Object state recognition
**Question**: Is the garbage bin open? \
**Program**:
```python
while not episode_is_over():
    explore_scene()
    object = detect_object('garbage bin')
    if object:
        navigate_to(object)
        answer('is the garbage bin open?')
        episode_is_over = True
```

## Spatial understanding
**Question**: Does the bedroom have a lot of furniture? \
**Program**:
```python   
while not episode_is_over():
    explore_scene()
    room = classify_room('bedroom')
    if room:
        look_around()
        answer(image, 'Does the bedroom have a lot of furniture?')
        episode_is_over = True
```

## Functional reasoning
**Question**: Where can I take a nap? \
**Program**:
```python   
while not episode_is_over():
    explore_scene()
    object = detect_object('bed')
    if object:
        navigate_to(object)
        answer('is the bed comfortable?')
        episode_is_over = True
```
## Object Localization
**Question**: Where is the blue trash can?  \
**Program**:
```python   
while not episode_is_over():
    explore_scene()
    object = detect_object('blue trash can')
    if object:
        navigate_to(object)
        look_around()
        answer('which room is this?')
        episode_is_over = True
```

## World knowledge
**Question**: There's something written on the whiteboard, what does it look to you?  \
**Program**:
```python   
while not episode_is_over():
    explore_scene()
    object = detect_object('whiteboard')
    if object:
        navigate_to(object)
        answer('what is written on the whiteboard?')
        episode_is_over = True
```

