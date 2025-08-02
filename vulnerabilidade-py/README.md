#Principais Vulnerabilidades em Aplicações Web

Este documento descreve de forma simples e objetiva algumas das vulnerabilidades mais comuns em aplicações web, com base em conceitos vistos no curso e no **OWASP Top 10**.

---

## 1) SQL Injection
**O que é:**  
Quando comandos SQL são injetados em campos de entrada que não têm validação adequada.

**Impacto:**  
- Acesso não autorizado a dados
- Modificação ou exclusão de registros
- Vazamento de informações sensíveis

## 2) Cross-Site Scripting (XSS)

**O que é:**
Permite que scripts maliciosos sejam injetados e executados no navegador de outros usuários.

**Impacto:**  

- Roubo de cookies ou sessões
- Redirecionamento para sites falsos
- Alteração da interface da página



## 3) Cross-Site Request Forgery (CSRF)

**O que é:**
Força um usuário autenticado a executar uma ação não desejada em uma aplicação.

**Impacto:**  

- Alteração de dados (ex: senha)
- Transferências bancárias não autorizadas
- Exclusão de informações importantes


##4) Quebra de autenticação

**O que é:**
Falhas nos mecanismos de login e autenticação que permitem acesso indevido.

**Impacto:**  

- Login como outro usuário
- Acesso a dados sensíveis ou funções administrativas


## 5) Exposição de dados sensíveis

**O que é:**
Falhas que permitem o vazamento de dados como senhas, cartões de crédito ou dados pessoais.

**Impacto:**  

- Vazamento público de informações
- Multas por não cumprimento de leis como LGPD/GDPR


