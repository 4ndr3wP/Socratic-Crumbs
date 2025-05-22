import React, { useRef, useEffect } from "react";

export default function UserWaveform({ audioData }) {
  const canvasRef = useRef();

  useEffect(() => {
    if (!audioData || audioData.length === 0) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = "#00ff00"; // bright green
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let i = 0; i < audioData.length; i++) {
      const x = (i / audioData.length) * canvas.width;
      const y = (1 - audioData[i]) * canvas.height / 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }, [audioData]);

  return <canvas ref={canvasRef} width={600} height={80} style={{ width: "100%", height: 80, background: "black" }} />;
} 