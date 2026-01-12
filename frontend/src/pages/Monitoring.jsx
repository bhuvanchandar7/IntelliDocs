import React, { useEffect, useState } from "react";
import { Activity, Server, Database, Cpu } from "lucide-react";
import { api } from "../api";

export default function Monitoring() {
    const [metrics, setMetrics] = useState({
        document_count: 0,
        total_requests: 0,
        average_latency_ms: 0,
        vector_db_status: "Checking..."
    });

    useEffect(() => {
        // Poll metrics every 5 seconds
        const fetchMetrics = async () => {
            const data = await api.getMetrics();
            if (data) setMetrics(data);
        };

        fetchMetrics();
        const interval = setInterval(fetchMetrics, 5000);
        return () => clearInterval(interval);
    }, []);

    const stats = [
        { title: "Avg. Query Latency", value: `${metrics.average_latency_ms}ms`, icon: Activity, trend: "Real-time", color: "text-green-500" },
        { title: "Vector DB Status", value: metrics.vector_db_status, icon: Database, trend: `${metrics.document_count} chunks`, color: "text-blue-500" },
        { title: "GPU Utilization", value: "N/A", icon: Cpu, trend: "CPU Mode", color: "text-purple-500" },
        { title: "Total Requests", value: metrics.total_requests, icon: Server, trend: "Global", color: "text-orange-500" },
    ];

    // Use real history or fallback to zeros if empty (initially)
    const history = metrics.request_history || new Array(60).fill(0);
    // Determine max value for scaling bar height (prevent flat line if max is 0)
    const maxVal = Math.max(...history, 5); 

    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div className="mb-8">
                <h1 className="text-3xl font-display font-bold text-slate-900 dark:text-white">System Monitoring</h1>
                <p className="text-slate-500 mt-1">Real-time performance metrics of your RAG pipeline.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                {stats.map((stat, i) => (
                    <div key={i} className="bg-white dark:bg-slate-900 p-6 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
                        <div className="flex items-center justify-between mb-4">
                            <div className={`p-3 rounded-xl bg-slate-50 dark:bg-slate-800 ${stat.color}`}>
                                <stat.icon className="w-6 h-6" />
                            </div>
                            <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400">
                                Live
                            </span>
                        </div>
                        <h3 className="text-3xl font-bold text-slate-900 dark:text-white mb-1">{stat.value}</h3>
                        <div className="flex items-center justify-between">
                            <p className="text-sm text-slate-500">{stat.title}</p>
                            <span className="text-xs font-medium text-slate-400">{stat.trend}</span>
                        </div>
                    </div>
                ))}
            </div>

            {/* Latency Chart Visualization */}
            <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-8 shadow-sm">
                <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">Request Volume (Last Hour)</h3>
                <div className="flex items-end justify-between h-48 gap-1">
                    {history.map((val, i) => (
                        <div
                            key={i}
                            className="flex-1 bg-primary/20 hover:bg-primary/60 transition-colors rounded-t-sm relative group"
                            style={{ height: `${(val / maxVal) * 100}%` }}
                        >
                            <div className="opacity-0 group-hover:opacity-100 absolute bottom-full left-1/2 -translate-x-1/2 mb-2 bg-slate-800 text-white text-xs py-1 px-2 rounded pointer-events-none whitespace-nowrap z-10">
                                {val} reqs
                            </div>
                        </div>
                    ))}
                </div>
                <div className="flex justify-between mt-4 text-xs text-slate-400">
                    <span>60 mins ago</span>
                    <span>Now</span>
                </div>
            </div>
        </div>
    );
}
