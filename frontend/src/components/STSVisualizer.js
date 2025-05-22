import React from "react";
import AssistantWaveform from "./AssistantWaveform";
import UserWaveform from "./UserWaveform";

export default function STSVisualizer({ assistantAudioData, userAudioData, assistantTranscript, userTranscript }) {
  return (
    <div style={{
      height: "100%",
      width: "100%",
      background: "black",
      display: "flex",
      flexDirection: "column",
      position: "relative"
    }}>
      <div style={{ flex: 1, borderBottom: "2px solid #222", position: "relative", display: 'flex', flexDirection: 'column', justifyContent: 'flex-start' }}>
        <AssistantWaveform audioData={assistantAudioData} />
        <div style={{
          color: '#a259ff', marginTop: 32, fontSize: 22, width: '100%', textAlign: 'center', fontWeight: 400
        }}>{assistantTranscript}</div>
      </div>
      <div style={{ flex: 1, position: "relative", display: 'flex', flexDirection: 'column', justifyContent: 'flex-start' }}>
        <UserWaveform audioData={userAudioData} />
        <div style={{
          color: '#fff', marginTop: 32, fontSize: 22, width: '100%', textAlign: 'center', fontWeight: 400
        }}>{userTranscript}</div>
      </div>
    </div>
  );
}