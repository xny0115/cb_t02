import React from 'react';
export interface Message {
  id: string;
  role: 'user' | 'bot';
  text: string;
}
export default function MessageItem({ role, text }: Message) {
  return <li className={role}>{text}</li>;
}
