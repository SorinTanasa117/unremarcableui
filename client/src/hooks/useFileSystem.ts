import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback } from 'react';

export function useFileSystem(watchInterval = 2000) {
  const queryClient = useQueryClient();
  const tree = useQuery({
    queryKey: ['file-tree'],
    queryFn: async () => {
      const res = await fetch('/api/files/tree');
      if (!res.ok) throw new Error('Failed to fetch file tree');
      return res.json();
    },
    refetchInterval: watchInterval > 0 ? watchInterval : false,
  });

  const readFile = async (path: string): Promise<string> => {
    const res = await fetch(`/api/files/read?path=${encodeURIComponent(path)}`);
    if (!res.ok) throw new Error('Failed to read file');
    const data = await res.json();
    return data.content;
  };

  const writeFile = async (path: string, content: string): Promise<void> => {
    const res = await fetch('/api/files/write', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, content }),
    });
    if (!res.ok) throw new Error('Failed to write file');
    queryClient.invalidateQueries({ queryKey: ['file-tree'] });
  };

  const deleteFile = async (path: string): Promise<void> => {
    const res = await fetch(`/api/files/delete?path=${encodeURIComponent(path)}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Failed to delete file');
    queryClient.invalidateQueries({ queryKey: ['file-tree'] });
  };

  // Hard refetch: bust the cache and force a network call. This always hits
  // the server even when the data is fresh, so the refresh button reliably
  // pulls newly created files / folders into the UI.
  const refetch = useCallback(() => {
    return queryClient.invalidateQueries({
      queryKey: ['file-tree'],
      refetchType: 'active',
    });
  }, [queryClient]);

  return {
    tree: tree.data ?? [],
    isLoading: tree.isLoading,
    isRefreshing: tree.isFetching,
    readFile,
    writeFile,
    deleteFile,
    refetch,
  };
}
