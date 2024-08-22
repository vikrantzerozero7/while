import module2

def main():
    
        query = st.text_input("Ask Question")
        # prompt: get question and answer part
        if st.button('Submit Answer'):
            result = chain.invoke(query)
        
        if "answer is not available in the context" in result:
            st.write("No answer")
        else:
            st.write(result)
            docs1 = vector_store.similarity_search(query,k=3)
            data_dict = docs1[0].metadata
            st.write("\nBook Name : ",data_dict["Book name"])
            st.write("Chapter : ",data_dict["Chapter"])
            st.write("Title : ",data_dict["Topic"])
            st.write("Subtopic : ",data_dict["Subtopic"])
            st.write("Subsubtopic : ",data_dict["Subsubtopic"])
    

if __name__=='__main__':
    main()
