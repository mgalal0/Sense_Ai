<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 800 700" xmlns="http://www.w3.org/2000/svg">
  <!-- Styles -->
  <style>
    .entity {
      fill: #e1f5fe;
      stroke: #0288d1;
      stroke-width: 2;
    }
    .entity-title {
      fill: #01579b;
      font-family: Arial, sans-serif;
      font-size: 14px;
      font-weight: bold;
    }
    .entity-attribute {
      fill: #333;
      font-family: Arial, sans-serif;
      font-size: 12px;
    }
    .relationship {
      stroke: #01579b;
      stroke-width: 2;
      fill: none;
    }
    .cardinality {
      fill: #01579b;
      font-family: Arial, sans-serif;
      font-size: 12px;
      font-weight: bold;
    }
    .pk {
      fill: #d50000;
    }
    .fk {
      fill: #6200ea;
    }
  </style>

  <!-- Entities -->
  
  <!-- EmotionAnalysis -->
  <rect class="entity" x="50" y="50" width="180" height="120" rx="5" ry="5" />
  <text x="140" y="75" text-anchor="middle" class="entity-title">EmotionAnalysis</text>
  <line x1="50" y1="80" x2="230" y2="80" stroke="#0288d1" stroke-width="1" />
  <text x="60" y="100" class="entity-attribute"><tspan class="pk">◆</tspan> id (PK)</text>
  <text x="60" y="120" class="entity-attribute">image</text>
  <text x="60" y="140" class="entity-attribute">emotion</text>
  <text x="60" y="160" class="entity-attribute">confidence</text>
  
  <!-- VideoAnalysis -->
  <rect class="entity" x="50" y="230" width="180" height="240" rx="5" ry="5" />
  <text x="140" y="255" text-anchor="middle" class="entity-title">VideoAnalysis</text>
  <line x1="50" y1="260" x2="230" y2="260" stroke="#0288d1" stroke-width="1" />
  <text x="60" y="280" class="entity-attribute"><tspan class="pk">◆</tspan> id (PK)</text>
  <text x="60" y="300" class="entity-attribute">video</text>
  <text x="60" y="320" class="entity-attribute">audio_file</text>
  <text x="60" y="340" class="entity-attribute">pdf_report</text>
  <text x="60" y="360" class="entity-attribute">dominant_emotion</text>
  <text x="60" y="380" class="entity-attribute">emotion_percentages</text>
  <text x="60" y="400" class="entity-attribute">emotion_durations</text>
  <text x="60" y="420" class="entity-attribute">total_frames</text>
  <text x="60" y="440" class="entity-attribute">video_duration</text>
  <text x="60" y="460" class="entity-attribute">transcription</text>
  
  <!-- RealTimeAnalysis -->
  <rect class="entity" x="330" y="50" width="180" height="160" rx="5" ry="5" />
  <text x="420" y="75" text-anchor="middle" class="entity-title">RealTimeAnalysis</text>
  <line x1="330" y1="80" x2="510" y2="80" stroke="#0288d1" stroke-width="1" />
  <text x="340" y="100" class="entity-attribute"><tspan class="pk">◆</tspan> id (PK)</text>
  <text x="340" y="120" class="entity-attribute">session_id</text>
  <text x="340" y="140" class="entity-attribute">start_time</text>
  <text x="340" y="160" class="entity-attribute">end_time</text>
  <text x="340" y="180" class="entity-attribute">is_active</text>
  <text x="340" y="200" class="entity-attribute">dominant_emotion</text>
  
  <!-- FrameAnalysis -->
  <rect class="entity" x="610" y="50" width="180" height="140" rx="5" ry="5" />
  <text x="700" y="75" text-anchor="middle" class="entity-title">FrameAnalysis</text>
  <line x1="610" y1="80" x2="790" y2="80" stroke="#0288d1" stroke-width="1" />
  <text x="620" y="100" class="entity-attribute"><tspan class="pk">◆</tspan> id (PK)</text>
  <text x="620" y="120" class="entity-attribute"><tspan class="fk">◇</tspan> session_id (FK)</text>
  <text x="620" y="140" class="entity-attribute">timestamp</text>
  <text x="620" y="160" class="entity-attribute">frame_number</text>
  <text x="620" y="180" class="entity-attribute">emotion</text>
  
  <!-- SentimentAnalysis -->
  <rect class="entity" x="330" y="230" width="180" height="120" rx="5" ry="5" />
  <text x="420" y="255" text-anchor="middle" class="entity-title">SentimentAnalysis</text>
  <line x1="330" y1="260" x2="510" y2="260" stroke="#0288d1" stroke-width="1" />
  <text x="340" y="280" class="entity-attribute"><tspan class="pk">◆</tspan> id (PK)</text>
  <text x="340" y="300" class="entity-attribute">text</text>
  <text x="340" y="320" class="entity-attribute">prediction</text>
  <text x="340" y="340" class="entity-attribute">sentiment</text>
  
  <!-- SpeechAnalysis -->
  <rect class="entity" x="330" y="370" width="180" height="140" rx="5" ry="5" />
  <text x="420" y="395" text-anchor="middle" class="entity-title">SpeechAnalysis</text>
  <line x1="330" y1="400" x2="510" y2="400" stroke="#0288d1" stroke-width="1" />
  <text x="340" y="420" class="entity-attribute"><tspan class="pk">◆</tspan> id (PK)</text>
  <text x="340" y="440" class="entity-attribute">audio</text>
  <text x="340" y="460" class="entity-attribute">transcription</text>
  <text x="340" y="480" class="entity-attribute">summary</text>
  <text x="340" y="500" class="entity-attribute">sentiment</text>
  
  <!-- TextSummary -->
  <rect class="entity" x="610" y="230" width="180" height="140" rx="5" ry="5" />
  <text x="700" y="255" text-anchor="middle" class="entity-title">TextSummary</text>
  <line x1="610" y1="260" x2="790" y2="260" stroke="#0288d1" stroke-width="1" />
  <text x="620" y="280" class="entity-attribute"><tspan class="pk">◆</tspan> id (PK)</text>
  <text x="620" y="300" class="entity-attribute">original_text</text>
  <text x="620" y="320" class="entity-attribute">summary</text>
  <text x="620" y="340" class="entity-attribute">min_length</text>
  <text x="620" y="360" class="entity-attribute">max_length</text>
  
  <!-- UserProfile -->
  <rect class="entity" x="330" y="530" width="180" height="100" rx="5" ry="5" />
  <text x="420" y="555" text-anchor="middle" class="entity-title">UserProfile</text>
  <line x1="330" y1="560" x2="510" y2="560" stroke="#0288d1" stroke-width="1" />
  <text x="340" y="580" class="entity-attribute"><tspan class="pk">◆</tspan> id (PK)</text>
  <text x="340" y="600" class="entity-attribute"><tspan class="fk">◇</tspan> user_id (FK)</text>
  <text x="340" y="620" class="entity-attribute">full_name</text>
  
  <!-- User (Django Auth) -->
  <rect class="entity" x="610" y="410" width="180" height="120" rx="5" ry="5" />
  <text x="700" y="435" text-anchor="middle" class="entity-title">User (Django Auth)</text>
  <line x1="610" y1="440" x2="790" y2="440" stroke="#0288d1" stroke-width="1" />
  <text x="620" y="460" class="entity-attribute"><tspan class="pk">◆</tspan> id (PK)</text>
  <text x="620" y="480" class="entity-attribute">username</text>
  <text x="620" y="500" class="entity-attribute">password</text>
  <text x="620" y="520" class="entity-attribute">email</text>
  
  <!-- Relationships -->
  
  <!-- RealTimeAnalysis to FrameAnalysis -->
  <path class="relationship" d="M510,130 H560 V130 H610" />
  <text x="530" y="125" class="cardinality">1</text>
  <text x="590" y="125" class="cardinality">N</text>
  
  <!-- User to UserProfile -->
  <path class="relationship" d="M700,530 V580 H510" />
  <text x="695" y="570" class="cardinality">1</text>
  <text x="530" y="575" class="cardinality">1</text>
  
  <!-- Legend -->
  <rect x="50" y="650" width="300" height="40" fill="#fff" stroke="#0288d1" stroke-width="1" />
  <text x="70" y="675" font-family="Arial, sans-serif" font-size="12">
    <tspan class="pk">◆</tspan> Primary Key (PK) | 
    <tspan class="fk">◇</tspan> Foreign Key (FK)
  </text>
</svg>