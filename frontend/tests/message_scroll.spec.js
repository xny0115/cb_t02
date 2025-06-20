try { require.resolve('react'); } catch { test.skip('React/Jest deps missing', () => {}); }
import React from 'react';
import { render } from '@testing-library/react';
import MessageList from '../src/components/MessageList';

describe('MessageList', () => {
  test('shows newest message first', () => {
    const messages = Array.from({ length: 30 }, (_, i) => ({
      id: String(i),
      role: 'bot',
      text: `msg${i}`,
    }));
    const { container } = render(React.createElement(MessageList, { messages }));
    const ul = container.firstElementChild;
    const first = ul && ul.firstElementChild;
    expect(first.textContent).toBe('msg29');
  });
});
