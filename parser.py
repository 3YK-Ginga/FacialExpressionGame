class SheetParser:
  def __init__(self, sheet):
    self.sheet = sheet
    self.notes = []
    self.speed = 5
    self.current_note_index = 0
    self.total_point = 0
    self.load_sheet()

  def load_sheet(self):
    try:
      with open(self.sheet, 'r', encoding='utf-8') as f:
        for line in f:
          line = line.strip()
          if line and not line.startswith('#'):
            parts = line.split(',')
            if len(parts) == 3:
              self.notes.append({
                'time': float(parts[0]),
                'emotion': parts[1],
                'point': int(parts[2])
              })
              self.total_point += int(parts[2])
      self.notes.sort(key=lambda x: x['time'])
    except Exception as e:
      print(f"譜面が読み込めません: {e}")
      self.notes = []
  
  def get_notes_at_time(self, load_time):
    notes_to_spawn = []
    while (self.current_note_index < len(self.notes) and self.notes[self.current_note_index]['time'] <= load_time):
      notes_to_spawn.append(self.notes[self.current_note_index])
      self.current_note_index += 1
    return notes_to_spawn
  
  def is_finished(self):
    return self.current_note_index >= len(self.notes)

  def reset(self):
    self.current_note_index = 0