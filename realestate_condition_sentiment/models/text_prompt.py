from typing import List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, IntEnum

# @dataclass
class PromptNeutrality(str, Enum):
  NEUTRAL = 'NEUTRAL'
  POSITIVE = 'POSITIVE'
  NEGATIVE = 'NEGATIVE'

class PromptGroupType(str, Enum):
  one_to_one = '1:1'
  one_to_many = '1:M'
  many = 'M'

class TaskType(str, Enum):
  FEATURE_DETECTION = 'FEATURE_DETECTION'
  QUALITY_SCORING = 'QUALITY_SCORING'
  SCENE_IDENTIFICATION = 'SCENE_IDENTIFICATION'

@dataclass
class TextPrompt:
  name: str = field(default='')
  text: str = field(default=None)
  neutrality: PromptNeutrality = PromptNeutrality.NEUTRAL

  def to_display(self):
    return self.text


@dataclass
class PromptGroup:
  name: str
  neutral_prompt: Union[TextPrompt, List[TextPrompt]]
  pos_prompt: List[TextPrompt] = field(default_factory=list)
  group_type: PromptGroupType = field(default=None)
  task_type: TaskType = field(default=None)    # the intended zero shot task type for this prompt(group)

  def __post_init__(self):
    if self.name is None:
      self.name = self.neutral_prompt.text
      
    if self.group_type is None:
      raise ValueError('group_type must be specified')
    
    if self.task_type is None:
      raise ValueError('task_type must be specified')
    
    if self.group_type == PromptGroupType.one_to_one:
      assert len(self.pos_prompt) == 1, 'one_to_one prompt group type must have only one pos_prompt'

    if isinstance(self.neutral_prompt, TextPrompt):
      self.neutral_prompt = [self.neutral_prompt]

  def to_dict(self):
    return {
      'name': self.name,
      'neutral_prompt': [p.__dict__ for p in self.neutral_prompt] if self.neutral_prompt is not None else None,
      'pos_prompt': [p.__dict__ for p in self.pos_prompt] if self.pos_prompt is not None else None,
      'group_type': self.group_type,
      'task_type': self.task_type
    }
  
  @classmethod
  def from_dict(cls, data):
    return cls(
            name=data['name'],
            neutral_prompt=cls._deserialize_neutral_prompt(data['neutral_prompt']),
            pos_prompt=cls._deserialize_pos_prompt(data['pos_prompt']),
            group_type=data['group_type'],
            task_type=data['task_type']
        )
  
  @staticmethod
  def _deserialize_neutral_prompt(data):
    if data is not None:
        return [TextPrompt(**prompt_data) for prompt_data in data]
    return None
  
  @staticmethod
  def _deserialize_pos_prompt(data):
    if data is not None:
        return [TextPrompt(**prompt_data) for prompt_data in data]
    return None
  
  def to_display(self):
    return {
        'name': self.name,
        'neutral_prompt': [p.to_display() for p in self.neutral_prompt] if self.neutral_prompt is not None else None,
        'pos_prompt': [p.to_display() for p in self.pos_prompt] if self.pos_prompt is not None else None,
        'group_type': self.group_type.value,
        'task_type': self.task_type.value
    }


kitchen_prompt_list = [

  PromptGroup(
    name='p_granite_countertop', 
    neutral_prompt=TextPrompt(text='a photo of a bathroom', neutrality=PromptNeutrality.NEUTRAL), 
    pos_prompt=[
        TextPrompt(text='a photo of a kitchen with beautiful granite counter top.', neutrality=PromptNeutrality.POSITIVE)
    ], 
    group_type=PromptGroupType.one_to_one,
    task_type=TaskType.FEATURE_DETECTION),

  PromptGroup(
    name='p_marble_countertop',
    neutral_prompt=TextPrompt(text='a photo of a kitchen', neutrality=PromptNeutrality.NEUTRAL),
    pos_prompt=[
        TextPrompt(text='a photo of a kitchen with beautiful marble counter top.', neutrality=PromptNeutrality.POSITIVE)
    ],
    group_type=PromptGroupType.one_to_one,
    task_type=TaskType.FEATURE_DETECTION),

  PromptGroup(
    name='p_quartz_countertop', 
    neutral_prompt=TextPrompt(text='a photo of a kitchen', neutrality=PromptNeutrality.NEUTRAL), 
    pos_prompt=[
        TextPrompt(text='a photo of a kitchen with beautiful quartz counter top.', neutrality=PromptNeutrality.POSITIVE)
    ], 
    group_type=PromptGroupType.one_to_one,
    task_type=TaskType.FEATURE_DETECTION),

  PromptGroup(
    name='p_kitchen_island',
    neutral_prompt=TextPrompt(text='a photo of a kitchen', neutrality=PromptNeutrality.NEUTRAL),
    pos_prompt=[
        TextPrompt(text='a photo of a kitchen with a large beautiful kitchen island.', neutrality=PromptNeutrality.POSITIVE)
    ],
    group_type=PromptGroupType.one_to_one,
    task_type=TaskType.FEATURE_DETECTION),

  PromptGroup(
    name='p_full_height_cabinets',
    neutral_prompt=TextPrompt(text='a photo of a kitchen', neutrality=PromptNeutrality.NEUTRAL),
    pos_prompt=[
        TextPrompt(text='a photo of a kitchen with beautiful full height cabinets.', neutrality=PromptNeutrality.POSITIVE)
    ],
    group_type=PromptGroupType.one_to_one,
    task_type=TaskType.FEATURE_DETECTION),

  PromptGroup(
    name='p_abundance_of_cabinet_storage',
    neutral_prompt=TextPrompt(text='a photo of a kitchen', neutrality=PromptNeutrality.NEUTRAL),
    pos_prompt=[
        TextPrompt(text='a photo of a kitchen with abundance of cabinet storage.', neutrality=PromptNeutrality.POSITIVE)        
    ],
    group_type=PromptGroupType.one_to_one,
    task_type=TaskType.FEATURE_DETECTION),

  PromptGroup(
    name='p_impressive_custom_kitchen_cabinetry',
    neutral_prompt=TextPrompt(text='a photo of a kitchen', neutrality=PromptNeutrality.NEUTRAL),
    pos_prompt=[
        TextPrompt(text='a photo of a kitchen with beautiful impressive custom kitchen cabinetry.', neutrality=PromptNeutrality.POSITIVE
        )
    ],
    group_type=PromptGroupType.one_to_one,
    task_type=TaskType.FEATURE_DETECTION),

  PromptGroup(
    name='p_recessed_lighting',
    neutral_prompt=TextPrompt(text='a photo of a kitchen', neutrality=PromptNeutrality.NEUTRAL),
    pos_prompt=[
        TextPrompt(text='a photo of a kitchen with beautiful recessed lighting.', neutrality=PromptNeutrality.POSITIVE)
    ],
    group_type=PromptGroupType.one_to_one,
    task_type=TaskType.FEATURE_DETECTION),

  PromptGroup(
    name='p_large_light_fixture_with_unique_finishes',
    neutral_prompt=TextPrompt(text='a photo of a kitchen', neutrality=PromptNeutrality.NEUTRAL),
    pos_prompt=[
        TextPrompt(
            text='a photo of a kitchen with large light fixture with unique finishes.',
            neutrality=PromptNeutrality.POSITIVE
        )
    ],
    group_type=PromptGroupType.one_to_one,
    task_type=TaskType.FEATURE_DETECTION),

  PromptGroup(
    name='p_excellent_kitchen',
    neutral_prompt=TextPrompt(text='a photo of a kitchen', neutrality=PromptNeutrality.NEUTRAL),
    pos_prompt=[
      TextPrompt(text='a photo of a beautiful gourmet kitchen.', neutrality=PromptNeutrality.POSITIVE),
      TextPrompt(text='a photo of a splendid dream kitchen.', neutrality=PromptNeutrality.POSITIVE),
      TextPrompt(text='a photo of a beautiful brightly lit kitchen.', neutrality=PromptNeutrality.POSITIVE),
      TextPrompt(text='a photo of a beautiful spacious kitchen.', neutrality=PromptNeutrality.POSITIVE),
      TextPrompt(text='a photo of a large stylish kitchen.', neutrality=PromptNeutrality.POSITIVE),
    ],
    group_type=PromptGroupType.one_to_many,
    task_type=TaskType.QUALITY_SCORING),

  PromptGroup(
    name='p_room',
    neutral_prompt=[
        TextPrompt(text='a photo of a bathroom.', neutrality=PromptNeutrality.NEUTRAL),
        TextPrompt(text='a photo of a kitchen.', neutrality=PromptNeutrality.NEUTRAL),
        TextPrompt(text='a photo of a living room.', neutrality=PromptNeutrality.NEUTRAL),
        # TextPrompt(text='a photo of a dining room.', neutrality=PromptNeutrality.NEUTRAL),
        TextPrompt(text='a photo of a bedroom.', neutrality=PromptNeutrality.NEUTRAL),
        TextPrompt(text='a photo of a garage.', neutrality=PromptNeutrality.NEUTRAL)
    ],
    pos_prompt=None,
    group_type=PromptGroupType.many,
    task_type=TaskType.SCENE_IDENTIFICATION),

  PromptGroup(
    name='p_indoor',
    neutral_prompt=TextPrompt(text='An outdoor photo', neutrality=PromptNeutrality.NEUTRAL),
    pos_prompt=[TextPrompt(text='An indoor photo', neutrality=PromptNeutrality.POSITIVE)],
    group_type=PromptGroupType.one_to_one,
    task_type=TaskType.SCENE_IDENTIFICATION)

]




