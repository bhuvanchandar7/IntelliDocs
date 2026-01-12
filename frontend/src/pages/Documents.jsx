import React, { useEffect, useState } from "react";
import { FileText, Plus, RefreshCw, Trash2, Loader2 } from "lucide-react";
import { api } from "../api";
import { cn } from "../utils";

// Simple Button Component
const Button = ({ children, className, variant = "default", size = "default", ...props }) => {
    const variants = {
        default: "bg-primary text-primary-foreground hover:bg-primary/90 shadow",
        outline: "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
        ghost: "hover:bg-accent hover:text-accent-foreground",
    };
    const sizes = {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-md px-3",
        lg: "h-11 rounded-md px-8",
        icon: "h-10 w-10",
    };

    return (
        <button
            className={cn(
                "inline-flex items-center justify-center rounded-lg text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
                variants[variant],
                sizes[size],
                className
            )}
            {...props}
        >
            {children}
        </button>
    );
};

export default function Documents() {
    const [documents, setDocuments] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [isUploading, setIsUploading] = useState(false);
    const fileInputRef = React.useRef(null);

    const fetchDocs = async () => {
        try {
            setIsLoading(true);
            const chunks = await api.getDocuments();

            const grouped = {};
            chunks.forEach(chunk => {
                const source = chunk.metadata.source || "Unknown";
                if (!grouped[source]) {
                    grouped[source] = {
                        id: chunk.id,
                        title: source.split('/').pop(),
                        source: source,
                        chunks: 0,
                    };
                }
                grouped[source].chunks += 1;
            });

            setDocuments(Object.values(grouped));
        } catch (e) {
            console.error(e);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchDocs();
    }, []);

    const handleUploadClick = () => {
        fileInputRef.current.click();
    };

    const handleFileChange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (file.type !== "application/pdf") {
            alert("Only PDF files are supported for now.");
            return;
        }

        try {
            setIsUploading(true);
            await api.uploadDocument(file);
            // alert("Document uploaded! Processing in background...");

            // Poll for updates (Background Task needs time to ingest)
            // We fetch immediately, then at 2s, 4s, 6s
            fetchDocs();
            setTimeout(fetchDocs, 2000);
            setTimeout(fetchDocs, 4000);
            setTimeout(fetchDocs, 6000);

        } catch (err) {
            alert("Upload failed: " + err.message);
        } finally {
            setIsUploading(false);
            e.target.value = null; // Reset input
        }
    };

    const handleDelete = async (source) => {
        if (!window.confirm(`Are you sure you want to delete ${source}?`)) return;

        try {
            await api.deleteDocument(source);
            fetchDocs(); // Refresh list
        } catch (err) {
            alert("Delete failed: " + err.message);
        }
    };

    const handleReprocess = (source) => {
        alert(`Reprocessing ${source} is not yet implemented (Requires backend job queue).`);
    };

    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            {/* Hidden File Input */}
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                className="hidden"
                accept="application/pdf"
            />

            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
                <div>
                    <h1 className="text-3xl font-display font-bold text-slate-900 dark:text-white">Knowledge Base</h1>
                    <p className="text-slate-500 mt-1">Manage documents used for AI context retrieval.</p>
                </div>

                <Button
                    size="lg"
                    className="rounded-xl shadow-lg shadow-primary/25"
                    onClick={handleUploadClick}
                    disabled={isUploading}
                >
                    {isUploading ? <Loader2 className="w-5 h-5 mr-2 animate-spin" /> : <Plus className="w-5 h-5 mr-2" />}
                    {isUploading ? "Uploading..." : "Add Document"}
                </Button>
            </div>

            {isLoading && !isUploading ? (
                <div className="flex items-center justify-center h-64">
                    <Loader2 className="w-8 h-8 text-primary animate-spin" />
                </div>
            ) : documents.length === 0 ? (
                <div className="text-center py-20 bg-slate-50 dark:bg-slate-900 rounded-2xl border-dashed border border-slate-200">
                    <p className="text-slate-500">No documents found in knowledge base.</p>
                </div>
            ) : (
                <div className="grid gap-4 max-h-[70vh] overflow-y-auto pr-2 custom-scrollbar">
                    {documents.map((doc) => (
                        <div
                            key={doc.id}
                            className="bg-white dark:bg-slate-900 p-5 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm hover:shadow-md transition-all group flex items-start justify-between"
                        >
                            <div className="flex items-start gap-4">
                                <div className="p-3 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 rounded-lg">
                                    <FileText className="w-6 h-6" />
                                </div>
                                <div>
                                    <h3 className="font-semibold text-slate-900 dark:text-white text-lg max-w-xs sm:max-w-md truncate" title={doc.source}>{doc.title}</h3>
                                    <div className="flex items-center gap-3 mt-1 text-sm text-slate-500">
                                        <span>{doc.chunks} chunks indexed</span>
                                    </div>
                                </div>
                            </div>

                            <div className="flex items-center gap-2 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 transition-opacity">
                                <Button variant="ghost" size="sm" className="text-slate-600" onClick={() => handleReprocess(doc.source)}>
                                    <RefreshCw className="w-4 h-4 mr-2" />
                                    Reprocess
                                </Button>
                                <Button variant="ghost" size="icon" className="text-slate-400 hover:text-red-500" onClick={() => handleDelete(doc.source)}>
                                    <Trash2 className="w-4 h-4" />
                                </Button>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

