# Automatizar Testes 

---

##**Objetivo**

- Explorar o uso de modelos de linguagem (LLMs) para gerar automaticamente casos de teste unitários.
- Integrar o LangChain como orquestrador da aplicação.
- Utilizar a API do Azure ChatGPT como provedor LLM.
- Documentar todo o processo técnico de forma clara.

---

##**Tecnologias e ferramentas utilizadas**

- Python 3.12+
- LangChain
- Azure OpenAI ChatGPT
- GitHub


---

##**Descrição do projeto**

O projeto cria um fluxo que:
1. Recebe como entrada um código Python sem testes.
2. Usa o LangChain para formatar a solicitação e interagir com o modelo de linguagem.
3. Envia o prompt para o Azure OpenAI ChatGPT.
4. Recebe do modelo um arquivo sugerido com testes unitários.
5. Salva esses testes em um arquivo `.py` separado para rodar via pytest.
