# Stage 1: Build the React Application
FROM node:18-alpine as build

WORKDIR /app

# Install Dependencies
COPY package*.json ./
RUN npm ci

# Copy Source Code
COPY . .

# Build for Production
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine

# Copy Build Artifacts from Stage 1
COPY --from=build /app/dist /usr/share/nginx/html

# Copy Custom Nginx Config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose Port 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
