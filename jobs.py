from dataclasses import dataclass

@dataclass
class Job:
    id: str
    title: str
    team: str
    location: str
    type: str          # Full-time / Part-time / Contract
    level: str         # Junior / Middle / Senior
    description: str
    responsibilities: list[str]
    requirements: list[str]
    skills: list[str]  # для NER matching
    nice_to_have: list[str]
    # English version of the vacancy text. Used for TF-IDF matching, because the
    # resumes and the TF-IDF vocabulary (tfidf.pkl) are English.
    text_en: str = ""

JOBS: dict[str, Job] = {

    "ai-engineer": Job(
        id="ai-engineer",
        title="AI / ML Engineer",
        team="AI Research",
        location="Remote",
        type="Full-time",
        level="Middle / Senior",
        description=(
            "Мы строим следующее поколение AI-продуктов и ищем инженера "
            "который умеет переводить исследования в production-системы. "
            "Вы будете работать с LLM, RAG-пайплайнами и MLOps-инфраструктурой."
        ),
        responsibilities=[
            "Разработка и деплой ML-моделей в production",
            "Построение RAG-пайплайнов на базе LLM",
            "Оптимизация inference: latency, cost, quality",
            "Написание чистого, тестируемого Python-кода",
            "Участие в code review и архитектурных обсуждениях",
        ],
        requirements=[
            "Python 3+ года",
            "Опыт с PyTorch или TensorFlow",
            "Понимание трансформеров и NLP",
            "Docker и базовый DevOps",
            "Опыт с векторными БД (Pinecone, Weaviate, Chroma)",
        ],
        skills=[
            "python", "pytorch", "tensorflow", "machine learning", "deep learning",
            "natural language processing", "docker", "aws", "fastapi",
            "langchain", "transformers", "mlflow", "kubernetes", "sql",
        ],
        nice_to_have=[
            "Опыт с LangChain / LlamaIndex",
            "Знание MLflow или аналогов",
            "Публикации или open-source вклад",
        ],
        text_en=(
            "AI / ML Engineer. Develop and deploy machine learning models to production. Build RAG pipelines with large language models (LLM). Optimize inference latency, cost and quality. Write clean, tested Python code and take part in code review. Requirements: 3+ years of Python, experience with PyTorch or TensorFlow, understanding of transformers and natural language processing (NLP), Docker and basic DevOps, vector databases (Pinecone, Weaviate, Chroma). Nice to have: LangChain or LlamaIndex, MLflow, open-source contributions."
        ),
    ),

    "backend": Job(
        id="backend",
        title="Backend Engineer",
        team="Platform",
        location="Remote / Hybrid",
        type="Full-time",
        level="Middle",
        description=(
            "Ищем backend-разработчика для работы над высоконагруженным API. "
            "Вы будете проектировать сервисы, оптимизировать запросы к БД "
            "и строить надёжную инфраструктуру."
        ),
        responsibilities=[
            "Разработка REST и GraphQL API",
            "Проектирование схем баз данных",
            "Оптимизация производительности запросов",
            "Написание unit и integration тестов",
            "Поддержка CI/CD пайплайнов",
        ],
        requirements=[
            "Python (FastAPI / Django) или Node.js",
            "PostgreSQL — уверенное владение",
            "Docker и docker-compose",
            "Понимание REST API и HTTP",
            "Git и базовый Linux",
        ],
        skills=[
            "python", "fastapi", "django", "node.js", "postgresql", "mongodb",
            "docker", "redis", "rest api", "graphql", "git", "linux",
            "ci/cd", "sql", "celery", "nginx",
        ],
        nice_to_have=[
            "Опыт с Kubernetes",
            "Знание Kafka или RabbitMQ",
            "Опыт с облачными провайдерами (AWS / GCP)",
        ],
        text_en=(
            "Backend Engineer. Develop REST and GraphQL APIs for a high-load service, design database schemas, optimize query performance, write unit and integration tests, maintain CI/CD pipelines. Requirements: Python (FastAPI, Django) or Node.js, strong PostgreSQL and SQL, Docker and docker-compose, REST API and HTTP, Git and Linux. Nice to have: Kubernetes, Kafka or RabbitMQ, Redis, Celery, Nginx, cloud providers (AWS, GCP)."
        ),
    ),

    "frontend": Job(
        id="frontend",
        title="Frontend Engineer",
        team="Product",
        location="Remote",
        type="Full-time",
        level="Middle",
        description=(
            "Ищем frontend-разработчика для создания современных "
            "пользовательских интерфейсов. Вы будете тесно работать "
            "с дизайнерами и backend-командой."
        ),
        responsibilities=[
            "Разработка UI-компонентов на React",
            "Интеграция с REST и GraphQL API",
            "Оптимизация производительности фронтенда",
            "Поддержка cross-browser совместимости",
            "Участие в проектировании UX",
        ],
        requirements=[
            "React + TypeScript — уверенное владение",
            "CSS / Sass — вёрстка без фреймворков",
            "REST API интеграция",
            "Git workflow",
            "Понимание Web Vitals и производительности",
        ],
        skills=[
            "javascript", "typescript", "react", "react.js", "next.js",
            "css", "sass", "html", "tailwind", "graphql", "rest api",
            "git", "webpack", "vite", "redux",
        ],
        nice_to_have=[
            "Опыт с Vue.js или Angular",
            "Знание Figma",
            "Опыт с тестированием (Jest, Cypress)",
        ],
        text_en=(
            "Frontend Engineer. Build modern user interfaces and UI components in React, integrate with REST and GraphQL APIs, optimize frontend performance, support cross-browser compatibility, take part in UX design. Requirements: strong React and TypeScript, JavaScript, HTML, CSS and Sass, REST API integration, Git workflow, Web Vitals. Nice to have: Next.js, Redux, Tailwind, Webpack or Vite, Vue.js or Angular, Figma, testing with Jest and Cypress."
        ),
    ),

    "data-engineer": Job(
        id="data-engineer",
        title="Data Engineer",
        team="Data Platform",
        location="Remote",
        type="Full-time",
        level="Middle / Senior",
        description=(
            "Строим data platform с нуля. Ищем инженера который умеет "
            "проектировать ETL-пайплайны, работать с большими объёмами данных "
            "и строить надёжные data warehouse решения."
        ),
        responsibilities=[
            "Проектирование и поддержка ETL/ELT пайплайнов",
            "Работа с потоковыми данными (Kafka, Spark)",
            "Оптимизация хранилищ данных",
            "Построение data quality мониторинга",
            "Документирование data lineage",
        ],
        requirements=[
            "Python — уверенное владение",
            "Apache Spark или PySpark",
            "SQL на продвинутом уровне",
            "Airflow или аналоги",
            "Опыт с облачными DWH (Redshift, BigQuery, Snowflake)",
        ],
        skills=[
            "python", "spark", "pyspark", "sql", "airflow", "kafka",
            "docker", "aws", "postgresql", "hadoop", "hive",
            "amazon redshift", "google bigquery", "dbt", "etl", "big data",
        ],
        nice_to_have=[
            "Опыт с dbt",
            "Знание Terraform",
            "Опыт с Databricks",
        ],
        text_en=(
            "Data Engineer. Build a data platform from scratch: design and maintain ETL/ELT pipelines, process streaming data with Kafka and Spark, optimize the data warehouse, build data quality monitoring and document data lineage. Requirements: strong Python, Apache Spark or PySpark, advanced SQL, Airflow, cloud data warehouses (Amazon Redshift, Google BigQuery, Snowflake), big data with Hadoop and Hive. Nice to have: dbt, Terraform, Databricks, Docker, AWS."
        ),
    ),

    "devops": Job(
        id="devops",
        title="DevOps / Platform Engineer",
        team="Infrastructure",
        location="Remote",
        type="Full-time",
        level="Middle / Senior",
        description=(
            "Ищем DevOps-инженера для построения и поддержки "
            "cloud-инфраструктуры. Вы будете отвечать за надёжность, "
            "масштабируемость и безопасность систем."
        ),
        responsibilities=[
            "Управление Kubernetes кластерами",
            "Построение CI/CD пайплайнов",
            "Infrastructure as Code (Terraform / Ansible)",
            "Мониторинг и алертинг (Prometheus, Grafana)",
            "Обеспечение безопасности инфраструктуры",
        ],
        requirements=[
            "Kubernetes — уверенное владение",
            "Docker и контейнеризация",
            "Terraform или Ansible",
            "AWS / GCP / Azure",
            "Linux и bash-scripting",
        ],
        skills=[
            "kubernetes", "docker", "terraform", "ansible", "aws", "gcp",
            "azure", "linux", "bash", "ci/cd", "python", "git",
            "nginx", "redis", "postgresql", "network security",
        ],
        nice_to_have=[
            "Опыт с Helm",
            "Знание Vault (HashiCorp)",
            "Опыт с eBPF или сетевой безопасностью",
        ],
        text_en=(
            "DevOps / Platform Engineer. Build and support cloud infrastructure: manage Kubernetes clusters, build CI/CD pipelines, infrastructure as code with Terraform and Ansible, monitoring and alerting with Prometheus and Grafana, infrastructure and network security. Requirements: strong Kubernetes, Docker and containerization, Terraform or Ansible, AWS, GCP or Azure, Linux and Bash scripting, Git, Python. Nice to have: Helm, HashiCorp Vault, Nginx, Redis, PostgreSQL."
        ),
    ),
}
