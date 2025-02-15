// import React, { useState, useEffect, useRef } from "react";
// import "./App.css";

// const SuggestionButton = ({ text, onClick }) => (
//   <button className="suggestion-button" onClick={() => onClick(text)}>
//     {text}
//   </button>
// );

// export default function App() {
//   const [query, setQuery] = useState("");
//   const [messages, setMessages] = useState([]);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState(null);
//   const messagesEndRef = useRef(null);

//   const suggestions = [
//     "What's your favorite sport?",
//     "Tell me a fun fact!",
//     "Recommend a good movie",
//     "Best travel destinations?",
//     "What's trending today?",
// ];


//   useEffect(() => {

//     const scrollToBottom = () => {
//       messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
//     };

//     scrollToBottom();
//   }, [messages]);

//   const sendQuery = async (inputQuery = query) => {
//     if (!inputQuery) return;

//     setLoading(true);
//     setError(null);
//     setMessages((prev) => [...prev, { content: inputQuery, isUser: true }]);
//     setQuery("");

//     try {
//       const res = await fetch("http://127.0.0.1:8000/query", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ query: inputQuery }),
//       });

//       if (!res.ok) {
//         throw new Error("Network response was not ok");
//       }

//       const data = await res.json();
//       setMessages((prev) => [...prev, { content: data.response, isUser: false }]);
//     } catch (err) {
//       console.error("Error fetching response:", err);
//       setError("Sorry, something went wrong. Please try again.");
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div className="app-container">
//       <div className="chat-container">
//         <div className="chat-header">
//           <div className="logo">
//             <div className="snowflake-icon"> ❄️ </div>
//             <div className="logo-text">MemoraNet</div>
//           </div>
//           <p className="chat-subtitle">Choose a prompt below or write your own to start chatting with Seam</p>

//           <p className="ask-about-text">Ask about:</p>
//         </div>

//         <div className="suggestions-container">
//           {suggestions.map((suggestion, index) => (
//             <SuggestionButton key={index} text={suggestion} onClick={sendQuery} />
//           ))}
//         </div>

//         <div className="messages-container">
//           {messages.map((message, index) => (
//             <div key={index} className={message.isUser ? "user-message" : "bot-message"}>
//               <div className="message-content">{message.content}</div>
//               <div className="message-timestamp">{new Date().toLocaleTimeString()}</div>
//             </div>
//           ))}
//           {loading && <div className="loading-spinner"></div>}
//           {error && <div className="error-message">{error}</div>}
//           <div ref={messagesEndRef} /> 
//         </div>

//         <div className="input-container">
//           <div className="input-wrapper">
//             <div className="input-icon">
//               <span role="img" aria-label="Prompt">
//                 🪄
//               </span>
//             </div>
//             <input
//               type="text"
//               className="chat-input"
//               placeholder="What are the best open opportunities by company size?"
//               value={query}
//               onChange={(e) => setQuery(e.target.value)}
//               onKeyPress={(e) => e.key === "Enter" && sendQuery()}
//             />
//           </div>

//           <div className="input-buttons">
//             <button className="send-button" onClick={() => sendQuery()} disabled={!query || loading}>
//               <svg className="send-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
//               <path 
//   strokeLinecap="round" 
//   strokeLinejoin="round" 
//   strokeWidth={2} 
//   d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" 
//   transform="rotate(90, 12, 12)"
// />
//               </svg>
//             </button>
//           </div>
//         </div>
//         <div className="character-count">0/2000</div>
//       </div>
//     </div>
//   );
// }

import React, { useState, useEffect, useRef } from "react";
import "./App.css";

const SuggestionButton = ({ text, onClick }) => (
  <button className="suggestion-button" onClick={() => onClick(text)}>
    {text}
  </button>
);

export default function App() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  const suggestions = [
    "What's your favorite sport?",
    "Tell me a fun fact!",
    "Recommend a good movie",
    "Best travel destinations?",
    "What's trending today?",
  ];

  useEffect(() => {
    const scrollToBottom = () => {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };
    scrollToBottom();
  }, [messages]);

  const sendQuery = async (inputQuery = query) => {
    if (!inputQuery) return;

    setLoading(true);
    setError(null);
    setMessages((prev) => [...prev, { content: inputQuery, isUser: true }]);
    setQuery("");

    try {
      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: inputQuery }),
      });

      if (!res.ok) {
        throw new Error(`Network response was not ok ${res.status} `);
      }

      const data = await res.json();
      setMessages((prev) => [...prev, { content: data.response, isUser: false }]);
    } catch (err) {
      console.error("Error fetching response:", err);
      setError("Sorry, something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="chat-container">
        <div className="chat-header">
          <div className="logo">
            <div className="snowflake-icon"> ❄️ </div>
            <div className="logo-text">MemoraNet</div>
          </div>
          <p className="chat-subtitle">Choose a prompt below or write your own to start chatting with Seam</p>
          <p className="ask-about-text">Ask about:</p>
        </div>

        <div className="suggestions-container">
          {suggestions.map((suggestion, index) => (
            <SuggestionButton key={index} text={suggestion} onClick={sendQuery} />
          ))}
        </div>

        <div className="messages-container">
          {messages.map((message, index) => (
            <div key={index} className={message.isUser ? "user-message" : "bot-message"}>
              <div className="message-content">{message.content}</div>
              <div className="message-timestamp">{new Date().toLocaleTimeString()}</div>
            </div>
          ))}
          {loading && <div className="loading-spinner"></div>}
          {error && <div className="error-message">{error}</div>}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <div className="input-wrapper">
            <div className="input-icon">
              <span role="img" aria-label="Prompt">
                🪄
              </span>
            </div>
            <input
              type="text"
              className="chat-input"
              placeholder="What are the best open opportunities by company size?"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && sendQuery()}
            />
          </div>
          <div className="input-buttons">
            <button className="send-button" onClick={() => sendQuery()} disabled={!query || loading}>
              <svg className="send-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" transform="rotate(90, 12, 12)" />
              </svg>
            </button>
          </div>
        </div>
        <div className="character-count">0/2000</div>
      </div>
    </div>
  );
}