import React from 'react';
import useTrainingStatus from '../hooks/useTrainingStatus';

export default function TrainingStatusBar() {
  const status = useTrainingStatus();
  if (!status.running) return null;
  return (
    <div className="bg-gray-800 text-xs text-white px-2 py-1">
      Epoch {status.current_epoch}/{status.total_epochs} – loss {status.loss.toFixed(4)}
    </div>
  );
}
