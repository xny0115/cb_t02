import React from 'react';

export default function MessageItem({ role, text }) {
  return <li className={role}>{text}</li>;
}
