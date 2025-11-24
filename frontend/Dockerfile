FROM node:lts-trixie-slim

WORKDIR /app

COPY package.json package-lock.json* ./

# Use npm install with legacy peer deps to avoid lockfile and peer resolution errors in build
RUN npm install --legacy-peer-deps

COPY . .

RUN npm run build

EXPOSE 5173

CMD ["npx", "vite", "preview", "--host", "0.0.0.0", "--port", "5173"]
