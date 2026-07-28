from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.core.deps import CurrentUser, LabelAccess, LabelRepoDep
from app.repo.base import Page, SortOrder
from app.repo.labels import Label, LabelCreate, LabelResponse, LabelSort, LabelUpdate

router = APIRouter(prefix="/labels", tags=["labels"])


@router.post("", response_model=LabelResponse, status_code=status.HTTP_201_CREATED)
async def create_label(
    body: LabelCreate, current_user: CurrentUser, labels: LabelRepoDep
) -> Label:
    label = Label(**body.model_dump(), created_by=current_user.id)
    return await labels.create(label)


@router.get("", response_model=Page[LabelResponse])
async def list_labels(
    current_user: CurrentUser,
    labels: LabelRepoDep,
    q: Annotated[str | None, Query(max_length=100)] = None,
    sort_by: LabelSort = LabelSort.NAME,
    order: SortOrder = SortOrder.ASC,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[LabelResponse]:
    items, total = await labels.list(
        current_user.id,
        q=q,
        sort_by=sort_by,
        order=order,
        limit=limit,
        offset=offset,
    )
    return Page[LabelResponse].model_validate(
        {"items": items, "total": total, "limit": limit, "offset": offset}
    )


@router.get("/{label_id}", response_model=LabelResponse)
async def get_label(
    label_id: LabelAccess, current_user: CurrentUser, labels: LabelRepoDep
) -> Label:
    label = await labels.get(label_id, current_user.id)
    if label is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Label not found")
    return label


@router.patch("/{label_id}", response_model=LabelResponse)
async def update_label(
    label_id: LabelAccess,
    body: LabelUpdate,
    current_user: CurrentUser,
    labels: LabelRepoDep,
) -> Label:
    label = await labels.update(
        label_id, current_user.id, body.model_dump(exclude_unset=True)
    )
    if label is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Label not found")
    return label


@router.delete("/{label_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_label(
    label_id: LabelAccess, current_user: CurrentUser, labels: LabelRepoDep
) -> None:
    if not await labels.delete(label_id, current_user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Label not found")
