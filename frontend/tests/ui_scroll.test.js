try { require.resolve('react'); } catch { test.skip('React/Jest deps missing', () => {}); }
require('@babel/register')({ presets: ['@babel/preset-react'] });
const React = require('react');
const { render } = require('@testing-library/react');
const MessageList = require('../src/components/MessageList').default;

test('shows newest message first', () => {
  const messages = Array.from({ length: 30 }, (_, i) => ({ id: String(i), role: 'bot', text: `msg${i}` })).reverse();
  const { container } = render(React.createElement(MessageList, { messages }));
  const first = container.firstChild.firstChild;
  expect(first.textContent).toBe('msg29');
});
