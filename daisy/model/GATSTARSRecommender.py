import torch
import torch.nn as nn
import torch.nn.functional as F

from daisy.model.AbstractRecommender import GeneralRecommender

class GATSTARS(GeneralRecommender):
    """
    STARS (PRISM) 모델의 DaisyRec 구현체입니다.

    Soft Topic Attention Recommender System (STARS), 또는
    Probabilistic Representation & Interpretation Space Model (PRISM)은
    사용자와 아이템을 k개의 잠재적 토픽(관심사)에 대한 프로필로 변환하고,
    두 프로필의 내적을 통해 추천 점수를 계산하는 Two-Tower 모델입니다.
    
    핵심 아이디어:
    1. 공유된 '토픽 키' (global_topic_keys)를 통해 사용자/아이템 임베딩을 토픽 프로필로 변환합니다.
    2. Softplus 활성화 함수를 사용하여 여러 토픽에 대한 동시적인 관심을 모델링합니다.
    3. 최종 점수는 두 토픽 프로필의 내적으로 계산되어, 높은 효율성과 확장성을 가집니다.
    """
    def __init__(self, config):
        """
        모델의 하이퍼파라미터와 레이어를 초기화합니다.
        
        Args:
            config (dict): 모델 설정을 담은 딕셔너리.
                           user_num, item_num, factors, num_topics, lr, reg_1, reg_2 등.
        """
        super(GATSTARS
        , self).__init__(config)

        # 하이퍼파라미터 로드
        self.factors = config['factors']
        self.num_topics = config['num_topics']
        self.num_heads = config['num_heads']
        self.lr = config['lr']
        self.reg_2 = config['reg_2']
        self.ortho_reg = config.get('ortho_reg', 0.01) # 직교 손실 가중치 추가
        self.topk = config['topk']

        # 임베딩 레이어 정의
        self.embed_user = nn.Embedding(config['user_num'], self.factors)
        self.embed_item = nn.Embedding(config['item_num'], self.factors)

        # STARS(PRISM)의 핵심 컴포넌트
        self.global_topic_keys = nn.Parameter(torch.randn(self.num_topics, self.factors))
        self.scale = nn.Parameter(torch.tensor(self.factors**-0.5))
        self.GATLayer = TopicGATLayer(self.factors, self.num_heads)
        #메인 손실함수
        self.ce_loss = nn.CrossEntropyLoss()
        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embed_user.weight)
        nn.init.xavier_uniform_(self.embed_item.weight)
        nn.init.orthogonal_(self.global_topic_keys.data)

    def _get_topic_profile(self, embeddings):
        """주어진 임베딩을 토픽 프로필로 변환합니다."""
        global_topic_keys = self.GATLayer(self.global_topic_keys)
        logits = torch.einsum('...d,kd->...k', embeddings, global_topic_keys) * self.scale
        profile = F.softplus(logits)
        return profile

    def forward(self, user, item):
        """사용자와 아이템 쌍에 대한 예측 점수를 계산합니다."""
        # 원본 임베딩
        user_embeddings = self.embed_user(user)
        item_embeddings = self.embed_item(item)

        # 토픽 프로필 변환
        user_topic_profile = self._get_topic_profile(user_embeddings)
        item_topic_profile = self._get_topic_profile(item_embeddings)
        
        # 내적을 통한 점수 계산
        # (batch, k) * (batch, k) -> (batch)
        pred = (user_topic_profile * item_topic_profile).sum(dim=1)
        
        return pred

    def calc_loss(self, batch):
        """
        Cross-Entropy Loss와 Orthogonal Loss를 결합하여 최종 손실을 계산합니다.
        
        Args:
            batch (tuple): (user, positive_item) 텐서 튜플.
                           DaisyRec의 CE loader는 positive item만 반환합니다.
        
        Returns:
            torch.Tensor: 계산된 총 손실.
        """
        user, pos_item = batch
        
        # 1. 주 손실 (Cross-Entropy Loss) 계산을 위한 전체 점수 계산
        user_emb = self.embed_user(user)
        user_topic_profile = self._get_topic_profile(user_emb) # (batch_size, k)

        all_item_embs = self.embed_item.weight
        all_item_topic_profiles = self._get_topic_profile(all_item_embs) # (num_items, k)
        
        all_scores = torch.matmul(user_topic_profile, all_item_topic_profiles.T) # (batch_size, num_items)

        # CE Loss 계산 (타겟은 positive item)
        loss_ce = self.ce_loss(all_scores, pos_item)

        # 2. 보조 손실 (Orthogonal Loss) 계산
        T = self.global_topic_keys
        k = self.num_topics
        
        # T * T^T가 항등 행렬(I)에 가까워지도록 강제
        identity = torch.eye(k, device=self.device)
        loss_ortho = torch.norm(torch.matmul(T, T.T) - identity)

        # 3. 최종 손실 결합
        total_loss = loss_ce + self.ortho_reg * loss_ortho

        return total_loss

    def predict(self, u, i):
        """특정 사용자와 아이템에 대한 예측을 수행합니다."""
        user = torch.tensor([u], device=self.device)
        item = torch.tensor([i], device=self.device)
        return self.forward(user, item)

    def rank(self, test_loader):
        """
        평가를 위한 전체 아이템 랭킹을 생성합니다.
        DaisyRec의 평가 루프에서 호출됩니다.
        """
        rec_ids_list = []
        for us, _, _ in test_loader:
                us = us.to(self.device)
                user_recs = self.full_rank(us)            # (batch, topk) Long
                rec_ids_list.append(user_recs)
        rec_ids = torch.cat(rec_ids_list, dim=0) 
        return rec_ids.cpu().numpy()

    def full_rank(self, u):
        """주어진 사용자에 대해 모든 아이템의 점수를 계산하고 Top-K를 반환합니다."""
        # 사용자 프로필 계산
        user_emb = self.embed_user(u)
        user_topic_profile = self._get_topic_profile(user_emb) # (batch_size, k)

        # 전체 아이템 프로필 계산
        all_item_embs = self.embed_item.weight
        all_item_topic_profiles = self._get_topic_profile(all_item_embs) # (num_items, k)
        
        # 전체 점수 계산 (행렬곱)
        scores = torch.matmul(user_topic_profile, all_item_topic_profiles.T) # (batch_size, num_items)
        
        # Top-K 아이템 ID 반환
        _, topk_ids = torch.topk(scores, self.topk, dim=1)

        return topk_ids
    
class TopicGATLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads=4):
        """
        불필요한 최적화를 줄이고 핵심 로직에 집중한 GAT 레이어.

        Args:
            embedding_dim (int): 임베딩 차원.
            num_heads (int): 어텐션 헤드 수 (기본값: 4).
        """
        super().__init__()
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 1. Q, K, V 프로젝션을 명시적으로 분리하여 가독성 증대
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        # 2. 핵심 안정화 장치
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, topic_keys):
        """
        Args:
            topic_keys (Tensor): 초기 토픽 키 (num_topics x embedding_dim).
        """
        K, d = topic_keys.shape
        residual = topic_keys

        # Q, K, V 생성 및 헤드 분할
        q = self.q_proj(topic_keys).view(K, self.num_heads, self.head_dim).transpose(0, 1) # (H, K, d_h)
        k = self.k_proj(topic_keys).view(K, self.num_heads, self.head_dim).transpose(0, 1) # (H, K, d_h)
        v = self.v_proj(topic_keys).view(K, self.num_heads, self.head_dim).transpose(0, 1) # (H, K, d_h)

        # 어텐션 스코어 및 가중치 계산
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.transpose(0, 1).contiguous().view(K, d)
        updated_topic_keys = self.layer_norm(residual + attention_output)
        
        return updated_topic_keys