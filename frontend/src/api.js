import axios from 'axios';

const API_URL = 'http://localhost:8000';

export const api = {
    healthCheck: async () => {
        try {
            const response = await axios.get(`${API_URL}/health`);
            return response.data;
        } catch (error) {
            console.error("Health check failed:", error);
            throw error;
        }
    },

    query: async (queryText) => {
        try {
            const response = await axios.post(`${API_URL}/query`, {
                query: queryText,
                limit: 3
            });
            return response.data;
        } catch (error) {
            console.error("Query failed:", error);
            throw error;
        }
    },

    getMetrics: async () => {
        try {
            const response = await axios.get(`${API_URL}/metrics`);
            return response.data;
        } catch (error) {
            console.error("Metrics fetch failed:", error);
            return null;
        }
    },

    getDocuments: async () => {
        try {
            const response = await axios.get(`${API_URL}/documents`);
            return response.data.documents;
        } catch (error) {
            console.error("Documents fetch failed:", error);
            return [];
        }
    },

    uploadDocument: async (file) => {
        const formData = new FormData();
        formData.append("file", file);
        try {
            const response = await axios.post(`${API_URL}/upload`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            return response.data;
        } catch (error) {
            console.error("Upload failed:", error);
            throw error;
        }
    },

    deleteDocument: async (source) => {
        try {
            const response = await axios.delete(`${API_URL}/documents`, {
                params: { source: source }
            });
            return response.data;
        } catch (error) {
            console.error("Delete failed:", error);
            throw error;
        }
    }
};
