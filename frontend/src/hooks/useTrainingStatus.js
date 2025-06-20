import { useEffect, useState } from 'react';

export default function useTrainingStatus() {
  const [status, setStatus] = useState({ running: false, current_epoch: 0, total_epochs: 0, loss: 0 });

  useEffect(() => {
    const fetchStatus = () => {
      fetch('/training/status')
        .then(r => r.json())
        .then(setStatus)
        .catch(() => {});
    };
    fetchStatus();
    const id = setInterval(fetchStatus, 2000);
    return () => clearInterval(id);
  }, []);

  return status;
}
