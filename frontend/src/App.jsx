import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, BookOpen, Loader2 } from 'lucide-react';
import { api } from './api';
import './index.css';

function App() {
    const [messages, setMessages] = useState([
        { role: 'system', content: 'Hello! I am IntelliDocs. Ask me anything about your documents.' }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMessage = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const data = await api.query(userMessage.content);

            const botMessage = {
                role: 'assistant',
                content: data.answer,
                sources: data.sources
            };

            setMessages(prev => [...prev, botMessage]);
        } catch (error) {
            setMessages(prev => [...prev, {
                role: 'system',
                content: 'Error: Could not retrieve answer. Make sure the backend is running.'
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex h-screen w-full bg-[#0f172a] text-slate-100">
            {/* Sidebar (Optional placeholder for history) */}
            <div className="w-64 hidden md:flex flex-col border-r border-slate-800 glass-panel p-4">
                <div className="flex items-center gap-2 mb-8 text-blue-400 font-bold text-xl">
                    <BookOpen /> IntelliDocs
                </div>
                <div className="text-sm text-slate-500 uppercase tracking-wider mb-2">History</div>
                <div className="flex-1 overflow-y-auto">
                    {/* History items would go here */}
                    <div className="p-2 rounded hover:bg-slate-800/50 cursor-pointer text-sm text-slate-400">
                        Transformer Architecture
                    </div>
                </div>
            </div>

            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col relative">
                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-4 md:p-8 gap-6 flex flex-col pb-32">
                    {messages.map((msg, idx) => (
                        <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>

                            {/* Avatar (Bot) */}
                            {msg.role !== 'user' && (
                                <div className="w-10 h-10 rounded-full bg-blue-600/20 flex items-center justify-center text-blue-400 flex-shrink-0">
                                    <Bot size={20} />
                                </div>
                            )}

                            {/* Message Bubble */}
                            <div className={`max-w-[80%] rounded-2xl p-4 shadow-lg ${msg.role === 'user'
                                    ? 'bg-blue-600 text-white rounded-br-none'
                                    : 'bg-slate-800/80 glass-panel rounded-bl-none border border-slate-700'
                                }`}>
                                <div className="whitespace-pre-wrap">{msg.content}</div>

                                {/* Citations/Sources */}
                                {msg.sources && msg.sources.length > 0 && (
                                    <div className="mt-4 pt-4 border-t border-slate-700/50">
                                        <div className="text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wide">Sources</div>
                                        <div className="flex flex-col gap-2">
                                            {msg.sources.map((source, i) => (
                                                <div key={i} className="bg-slate-900/50 p-2 rounded border border-slate-800 text-xs text-slate-300">
                                                    <div className="font-medium text-blue-400 mb-1">{source.source}</div>
                                                    <div className="line-clamp-2 opacity-80">{source.content}</div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Avatar (User) */}
                            {msg.role === 'user' && (
                                <div className="w-10 h-10 rounded-full bg-slate-700 flex items-center justify-center text-slate-300 flex-shrink-0">
                                    <User size={20} />
                                </div>
                            )}
                        </div>
                    ))}
                    {isLoading && (
                        <div className="flex gap-4">
                            <div className="w-10 h-10 rounded-full bg-blue-600/20 flex items-center justify-center text-blue-400">
                                <Bot size={20} />
                            </div>
                            <div className="bg-slate-800/80 glass-panel rounded-2xl p-4 flex items-center gap-2 text-slate-400">
                                <Loader2 className="animate-spin" size={16} /> Thinking...
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-[#0f172a] via-[#0f172a] to-transparent">
                    <div className="max-w-4xl mx-auto glass-panel rounded-xl p-2 flex gap-2 items-center border border-slate-700/50 shadow-2xl">
                        <input
                            type="text"
                            className="flex-1 bg-transparent border-none outline-none text-white px-4 py-3 placeholder-slate-500"
                            placeholder="Ask a question about your documents..."
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSend(e)}
                        />
                        <button
                            onClick={handleSend}
                            disabled={isLoading || !input.trim()}
                            className="p-3 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all text-white shadow-lg shadow-blue-900/20"
                        >
                            <Send size={20} />
                        </button>
                    </div>
                    <div className="text-center text-xs text-slate-600 mt-2">
                        IntelliDocs v1.0 • Powered by Mistral-7B & RAG
                    </div>
                </div>
            </div>
        </div>
    );
}

export default App;
