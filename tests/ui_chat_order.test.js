try { require.resolve('react'); } catch { test.skip('React/Jest deps missing', () => {}); }
require('@babel/register')({ presets: ['@babel/preset-react'] });
const React = require('react');
const { render } = require('@testing-library/react');
const MessageList = require('../frontend/src/components/MessageList').default;

test('firstChild is newest message', () => {
  const msgs = Array.from({ length: 5 }, (_, i) => ({ id: String(i), role: 'bot', text: `m${i}` })).reverse();
  const { container } = render(React.createElement(MessageList, { messages: msgs }));
  expect(container.firstChild.firstChild.textContent).toBe('m4');
});
