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
    }
};
