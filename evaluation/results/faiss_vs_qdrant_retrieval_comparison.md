# FAISS vs Qdrant Retrieval Comparison

This report compares the Phase 1 text retriever over the same IFC Annual Report embeddings.

| Query | Top-k overlap | FAISS pages | Qdrant pages |
| --- | ---: | --- | --- |
| What is IFC's mission? | 5 | [5, 5, 12, 8, 8] | [5, 5, 12, 8, 8] |
| What was the net income for FY24 and FY23? | 5 | [44, 5, 44, 50, 8] | [44, 5, 44, 50, 8] |
| Show me the trend of IFC's net income from FY22 to FY24. | 5 | [6, 44, 5, 8, 44] | [6, 44, 5, 8, 44] |

## Retrieved Evidence

### What is IFC's mission?

FAISS:
- rank 1, page 5, text: open only to member countries of IBRD. IFC is not liable for the obligations of the other institutions. IFC’s mission — as one of the WBG entities — is to end extreme poverty and boost shared prosperity on a livable planet. As the private s
- rank 2, page 5, text: SECTION l. EXECUTIVE SUMMARY This executive summary highlights selected informa - tion and may not contain all of the information that is important to readers of this document. For a complete description of IFC’s FY24 performance, as well a
- rank 3, page 12, text: the objective to build and proactively manage a portfolio that produces strong financial results and development impact. IFC achieves this through a combination of strong presence on the ground and deep sector expertise, that enables IFC to
- rank 4, page 8, text: SECTION II. OVERVIEW IFC is the largest global development institution focused on the private sector in developing countries. Established in 1956, IFC is owned by 186 member countries, a group that collectively determines its policies. IFC 
- rank 5, page 8, text: services to businesses and governments. IFC’s principal investment products are loans, equity investments, debt securities and guarantees. IFC also plays an active and direct role in mobilizing additional funding from other investors and le

Qdrant:
- rank 1, page 5, text: open only to member countries of IBRD. IFC is not liable for the obligations of the other institutions. IFC’s mission — as one of the WBG entities — is to end extreme poverty and boost shared prosperity on a livable planet. As the private s
- rank 2, page 5, text: SECTION l. EXECUTIVE SUMMARY This executive summary highlights selected informa - tion and may not contain all of the information that is important to readers of this document. For a complete description of IFC’s FY24 performance, as well a
- rank 3, page 12, text: the objective to build and proactively manage a portfolio that produces strong financial results and development impact. IFC achieves this through a combination of strong presence on the ground and deep sector expertise, that enables IFC to
- rank 4, page 8, text: SECTION II. OVERVIEW IFC is the largest global development institution focused on the private sector in developing countries. Established in 1956, IFC is owned by 186 member countries, a group that collectively determines its policies. IFC 
- rank 5, page 8, text: services to businesses and governments. IFC’s principal investment products are loans, equity investments, debt securities and guarantees. IFC also plays an active and direct role in mobilizing additional funding from other investors and le

### What was the net income for FY24 and FY23?

FAISS:
- rank 1, page 44, text: IFC’s net income or loss for the past three fiscal years ended June 30, 2024 are presented below: Figure 25: IFC's Net Income (Loss) FY22–FY24 (US$ in millions) -1,000 0 500-500 (464) 672 1,000 1,500 2,000 Fiscal year ended June 30, FY22 FY
- rank 2, page 5, text: guarantee issuances for all entities. FINANCIAL PERFORMANCE SUMMARY IFC’s financial performance has been influenced by the changes in interest rates in FY24. NET INCOME AND INCOME AVAILABLE FOR DESIGNATIONS IFC’s net income was $1.5 billion
- rank 3, page 44, text: Others*** Change in Net Income Total Income from Loans and Debt Securities* * Total income from loans and debt securities and net treasury income are net of allocated charges on borrowings. ** URG(L) refers to Unrealized Gains (Losses). ***
- rank 4, page 50, text: Reclassification adjustment for realized gains included in net income upon derecognition of borrowings 12 12 Net unrealized gains (losses) on borrowings $ 74 $ (50) Total unrealized gains on debt securities and borrowings $ 195 $ 12 Net unr
- rank 5, page 8, text: of the IDA Eighteen Replenishment of Resources (IDA18). In FY24, IFC updated the calculation of Income Available for Designations to exclude income from Post-retirement Contribution Reserve Fund (PCRF), aligning it with its intended use for

Qdrant:
- rank 1, page 44, text: IFC’s net income or loss for the past three fiscal years ended June 30, 2024 are presented below: Figure 25: IFC's Net Income (Loss) FY22–FY24 (US$ in millions) -1,000 0 500-500 (464) 672 1,000 1,500 2,000 Fiscal year ended June 30, FY22 FY
- rank 2, page 5, text: guarantee issuances for all entities. FINANCIAL PERFORMANCE SUMMARY IFC’s financial performance has been influenced by the changes in interest rates in FY24. NET INCOME AND INCOME AVAILABLE FOR DESIGNATIONS IFC’s net income was $1.5 billion
- rank 3, page 44, text: Others*** Change in Net Income Total Income from Loans and Debt Securities* * Total income from loans and debt securities and net treasury income are net of allocated charges on borrowings. ** URG(L) refers to Unrealized Gains (Losses). ***
- rank 4, page 50, text: Reclassification adjustment for realized gains included in net income upon derecognition of borrowings 12 12 Net unrealized gains (losses) on borrowings $ 74 $ (50) Total unrealized gains on debt securities and borrowings $ 195 $ 12 Net unr
- rank 5, page 8, text: of the IDA Eighteen Replenishment of Resources (IDA18). In FY24, IFC updated the calculation of Income Available for Designations to exclude income from Post-retirement Contribution Reserve Fund (PCRF), aligning it with its intended use for

### Show me the trend of IFC's net income from FY22 to FY24.

FAISS:
- rank 1, page 6, text: Figure 1: Income Measures (US$ in millions) Income Available for Designations/uni00A0/uni00A0/uni00A0 Net income (loss) FY24FY23 0 500 1,000 1,500 2,000 FY22 (500) 681 672 382 (464) 1,558 1,485 INVESTMENT OPERATIONS In FY24, IFC delivered a
- rank 2, page 44, text: IFC’s net income or loss for the past three fiscal years ended June 30, 2024 are presented below: Figure 25: IFC's Net Income (Loss) FY22–FY24 (US$ in millions) -1,000 0 500-500 (464) 672 1,000 1,500 2,000 Fiscal year ended June 30, FY22 FY
- rank 3, page 5, text: guarantee issuances for all entities. FINANCIAL PERFORMANCE SUMMARY IFC’s financial performance has been influenced by the changes in interest rates in FY24. NET INCOME AND INCOME AVAILABLE FOR DESIGNATIONS IFC’s net income was $1.5 billion
- rank 4, page 8, text: of the IDA Eighteen Replenishment of Resources (IDA18). In FY24, IFC updated the calculation of Income Available for Designations to exclude income from Post-retirement Contribution Reserve Fund (PCRF), aligning it with its intended use for
- rank 5, page 44, text: Others*** Change in Net Income Total Income from Loans and Debt Securities* * Total income from loans and debt securities and net treasury income are net of allocated charges on borrowings. ** URG(L) refers to Unrealized Gains (Losses). ***

Qdrant:
- rank 1, page 6, text: Figure 1: Income Measures (US$ in millions) Income Available for Designations/uni00A0/uni00A0/uni00A0 Net income (loss) FY24FY23 0 500 1,000 1,500 2,000 FY22 (500) 681 672 382 (464) 1,558 1,485 INVESTMENT OPERATIONS In FY24, IFC delivered a
- rank 2, page 44, text: IFC’s net income or loss for the past three fiscal years ended June 30, 2024 are presented below: Figure 25: IFC's Net Income (Loss) FY22–FY24 (US$ in millions) -1,000 0 500-500 (464) 672 1,000 1,500 2,000 Fiscal year ended June 30, FY22 FY
- rank 3, page 5, text: guarantee issuances for all entities. FINANCIAL PERFORMANCE SUMMARY IFC’s financial performance has been influenced by the changes in interest rates in FY24. NET INCOME AND INCOME AVAILABLE FOR DESIGNATIONS IFC’s net income was $1.5 billion
- rank 4, page 8, text: of the IDA Eighteen Replenishment of Resources (IDA18). In FY24, IFC updated the calculation of Income Available for Designations to exclude income from Post-retirement Contribution Reserve Fund (PCRF), aligning it with its intended use for
- rank 5, page 44, text: Others*** Change in Net Income Total Income from Loans and Debt Securities* * Total income from loans and debt securities and net treasury income are net of allocated charges on borrowings. ** URG(L) refers to Unrealized Gains (Losses). ***
