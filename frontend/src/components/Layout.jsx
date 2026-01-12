import { Link, useLocation } from "wouter";
import { MessageSquare, FileText, Activity, Menu, X } from "lucide-react";
import React, { useState } from "react";
import { cn } from "../utils";

export function Layout({ children }) {
    const [location] = useLocation();
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

    const navItems = [
        { href: "/", label: "Chat", icon: MessageSquare },
        { href: "/documents", label: "Documents", icon: FileText },
        { href: "/monitoring", label: "Monitoring", icon: Activity },
    ];

    return (
        <div className="flex h-screen bg-slate-50 dark:bg-slate-950 overflow-hidden font-sans text-slate-900 dark:text-slate-50">
            {/* Sidebar - Desktop */}
            <aside className="hidden md:flex w-64 flex-col border-r border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 z-20">
                <div className="p-6 border-b border-slate-100 dark:border-slate-800">
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-accent flex items-center justify-center text-white font-bold shadow-lg shadow-primary/20">
                            ID
                        </div>
                        <h1 className="text-xl font-display font-bold text-slate-900 dark:text-white tracking-tight">
                            IntelliDocs
                        </h1>
                    </div>
                </div>

                <nav className="flex-1 p-4 space-y-1">
                    {navItems.map((item) => {
                        const isActive = location === item.href || (location === "/" && item.href === "/");
                        return (
                            <Link key={item.href} href={item.href}>
                                <div className={cn(
                                    "flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 cursor-pointer group",
                                    isActive
                                        ? "bg-primary/10 text-primary font-medium shadow-sm"
                                        : "text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-100"
                                )}>
                                    <item.icon className={cn("w-5 h-5", isActive ? "text-primary" : "text-slate-400 group-hover:text-slate-600 dark:group-hover:text-slate-300")} />
                                    {item.label}
                                </div>
                            </Link>
                        );
                    })}
                </nav>

                <div className="p-4 border-t border-slate-100 dark:border-slate-800">
                    <div className="p-4 rounded-xl bg-gradient-to-br from-slate-900 to-slate-800 text-white shadow-lg">
                        <p className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-1">System Status</p>
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                            <span className="text-sm font-semibold">Operational</span>
                        </div>
                    </div>
                </div>
            </aside>

            {/* Mobile Header */}
            <div className="md:hidden fixed top-0 left-0 right-0 h-16 bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 z-50 flex items-center justify-between px-4">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center text-white font-bold">
                        ID
                    </div>
                    <span className="font-bold text-lg">IntelliDocs</span>
                </div>
                <button onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)} className="p-2">
                    {isMobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
                </button>
            </div>

            {/* Mobile Menu */}
            {isMobileMenuOpen && (
                <div className="md:hidden fixed inset-0 bg-white dark:bg-slate-900 z-40 pt-20 px-4 space-y-2">
                    {navItems.map((item) => (
                        <Link key={item.href} href={item.href}>
                            <div
                                className="flex items-center gap-3 px-4 py-4 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer border-b border-slate-100 dark:border-slate-800"
                                onClick={() => setIsMobileMenuOpen(false)}
                            >
                                <item.icon className="w-5 h-5" />
                                {item.label}
                            </div>
                        </Link>
                    ))}
                </div>
            )}

            {/* Main Content */}
            <main className="flex-1 overflow-auto md:pt-0 pt-16 relative">
                <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 pointer-events-none mix-blend-soft-light"></div>
                {children}
            </main>
        </div>
    );
}
