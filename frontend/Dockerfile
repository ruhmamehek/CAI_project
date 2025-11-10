# Build stage
FROM node:18-alpine AS build

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source files
COPY . .

# Build the application
RUN npm run build

# Production stage - use Vite preview
FROM node:18-alpine

WORKDIR /app

# Copy package files and node_modules from build stage
COPY --from=build /app/package*.json ./
COPY --from=build /app/node_modules ./node_modules

# Copy built files
COPY --from=build /app/dist ./dist

# Expose port
EXPOSE 3000

# Use Vite's preview server
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "3000"]

