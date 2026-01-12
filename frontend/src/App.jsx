import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { Send, Bot, User } from 'lucide-react';
import { Switch, Route } from "wouter";
import { cn } from './utils';
import { api } from './api';

// Pages & Components
import { Layout } from './components/Layout';
import Documents from './pages/Documents';
import Monitoring from './pages/Monitoring';

// --- Chat Component (Refactored) ---
function Chat() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
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
        if (!input.trim() || isLoading) return;

        const userMessage = { id: Date.now(), role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput("");
        setIsLoading(true);

        try {
            const response = await api.query(userMessage.content);

            const botMessage = {
                id: Date.now() + 1,
                role: 'assistant',
                content: response.answer,
                sources: response.sources
            };

            setMessages(prev => [...prev, botMessage]);
        } catch (error) {
            console.error("Error:", error);
            const errorMessage = {
                id: Date.now() + 1,
                role: 'assistant',
                content: "Sorry, I encountered an error while processing your request."
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full bg-white dark:bg-slate-950">
            {/* Messages List */}
            <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6 scroll-smooth">
                {messages.length === 0 ? (
                    <div className="flex-1 flex flex-col items-center justify-center text-center p-8 mt-10 sm:mt-20">
                        <div className="w-20 h-20 bg-gradient-to-br from-primary/20 to-accent/20 rounded-3xl flex items-center justify-center mb-6 shadow-xl shadow-primary/10">
                            <Bot className="w-10 h-10 text-primary" />
                        </div>
                        <h2 className="text-3xl font-display font-bold text-slate-900 dark:text-white mb-3">
                            How can I help you today?
                        </h2>
                        <p className="text-slate-500 max-w-md mb-8">
                            I can analyze your uploaded documents and answer questions based on their content.
                        </p>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-2xl w-full">
                            {["Summarize the key findings", "What is the methodology?", "Explain the conclusion"].map((suggestion) => (
                                <button
                                    key={suggestion}
                                    onClick={() => { setInput(suggestion); }}
                                    className="p-4 text-left rounded-xl border border-slate-200 dark:border-slate-800 hover:border-primary/50 hover:bg-slate-50 dark:hover:bg-slate-900 transition-all group"
                                >
                                    <span className="text-sm font-medium text-slate-700 dark:text-slate-300 group-hover:text-primary transition-colors">{suggestion}</span>
                                </button>
                            ))}
                        </div>
                    </div>
                ) : (
                    messages.map((msg) => (
                        <div
                            key={msg.id}
                            className={cn(
                                "flex gap-4 max-w-4xl mx-auto",
                                msg.role === 'user' ? "flex-row-reverse" : "flex-row"
                            )}
                        >
                            <div className={cn(
                                "w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 shadow-sm",
                                msg.role === 'user' ? "bg-primary text-white" : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300"
                            )}>
                                {msg.role === 'user' ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
                            </div>

                            <div className={cn(
                                "flex-1 px-5 py-3.5 rounded-2xl shadow-sm text-sm leading-relaxed",
                                msg.role === 'user'
                                    ? "bg-primary text-primary-foreground rounded-tr-sm"
                                    : "bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-tl-sm"
                            )}>
                                <div className="prose-custom break-words whitespace-pre-wrap font-medium">
                                    {msg.content}
                                </div>

                                {/* Sources Section */}
                                {msg.sources && msg.sources.length > 0 && (
                                    <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-800">
                                        <p className="text-xs font-semibold text-slate-500 mb-2">Sources:</p>
                                        <div className="grid grid-cols-1 gap-2">
                                            {msg.sources.map((source, idx) => (
                                                <div key={idx} className="bg-slate-50 dark:bg-slate-950 p-2 rounded border border-slate-200 dark:border-slate-800 text-xs text-slate-600 dark:text-slate-400">
                                                    <span className="font-medium text-primary block mb-1">{source.source}</span>
                                                    {source.content.substring(0, 150)}...
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    ))
                )}

                {isLoading && (
                    <div className="flex gap-4 max-w-4xl mx-auto">
                        <div className="w-8 h-8 rounded-lg bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 flex items-center justify-center flex-shrink-0">
                            <Bot className="w-5 h-5 text-slate-400" />
                        </div>
                        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 px-5 py-4 rounded-2xl rounded-tl-sm shadow-sm flex items-center gap-1">
                            <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                            <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                            <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 bg-white dark:bg-slate-950 border-t border-slate-200 dark:border-slate-800">
                <div className="max-w-4xl mx-auto relative">
                    <form onSubmit={handleSend} className="relative flex items-center">
                        <input
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ask a question about your documents..."
                            className="w-full pr-12 pl-4 py-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 focus-visible:outline-none focus:ring-2 focus:ring-primary/20 shadow-sm transition-all"
                            disabled={isLoading}
                        />
                        <button
                            type="submit"
                            disabled={!input.trim() || isLoading}
                            className="absolute right-2 top-1/2 -translate-y-1/2 h-10 w-10 flex items-center justify-center bg-primary hover:bg-primary/90 text-white rounded-lg transition-all disabled:opacity-50"
                        >
                            <Send className="w-4 h-4" />
                        </button>
                    </form>
                    <p className="text-center text-xs text-slate-400 mt-2">
                        AI can make mistakes. Verify important information.
                    </p>
                </div>
            </div>
        </div>
    );
}

// --- Main App with Routing ---
function App() {
    return (
        <Layout>
            <Switch>
                <Route path="/" component={Chat} />
                <Route path="/documents" component={Documents} />
                <Route path="/monitoring" component={Monitoring} />
                <Route>
                    {/* Fallback 404 */}
                    <div className="flex items-center justify-center h-full text-slate-500">
                        Page not found
                    </div>
                </Route>
            </Switch>
        </Layout>
    );
}

export default App;
