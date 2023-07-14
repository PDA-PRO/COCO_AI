position_idx=[2,3,4,5,6,7,8,9,0,0,0,0,0,0,0,0,0]
nodes_mask=position_idx.eq(0)
token_mask=position_idx.ge(2)     
print(nodes_mask)
print(token_mask)   
# inputs_embeddings=self.encoder.embeddings.word_embeddings(source_ids)
# nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
# nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
# avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)